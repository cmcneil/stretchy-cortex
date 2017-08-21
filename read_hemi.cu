extern "C" {
  #include <gifti_io.h>
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <list>

#include <GL/glew.h>
#include <glfw3.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>


// CUDA helper functions
// #include <helper_cuda.h>
// #include <helper_cuda_gl.h>

#include "shader.h"
#include "controls.h"

// using namespace glm;
using namespace std;
using std::vector;
using std::list;
using glm::vec3;
// using thrust::device_vector;
// using thrust::host_vector;

template <typename T>
struct devarray {
    size_t nelements;
    T *data;
};

__device__
void updateVertexPositionVerlet(vec3 *pos, vec3 *prev, vec3 force) {
  vec3 temp = *pos;
  float damping = 0.0f;
  float tstep = 0.1f;
  *pos = temp + (temp - *prev) * (1.0f - damping) + force * tstep;
  *prev = temp;
}

__device__
vec3 getAccelerationOnVertex(vec3 *pos, int idx, devarray neighbor_list) {
  vec3 acceleration = vec3(0.0f, 0.0f, 0.0f);
  vec3 mypos = pos[idx];
  float k = 0.01;
  for (int i=0; i < neighbor_list.nelements; i++) {
    vec3 nbr_pos = pos[neighbor_list.data[i]];
    float dist = glm::distance(nbr_pos, mypos);
    vec3 dir_to_nbr = glm::normalize(nbr_pos - mypos);
    acceleration += dir_to_nbr*dist*k;
  }
  return acceleration;
}

__global__
void simple_distortion(vec3 *pos, int total_pts, float t, devarray* adj_list,
                       vec3 *prevs) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    float pi = acosf(-1);

    if (idx >= total_pts) {
      return;
    }

    float period = 10.0f; //s
    float w = sinf(2*pi*(t + __int2float_rd(idx) / __int2float_rd(total_pts))/period) * 0.01f;

    vec3 accel = getAccelerationOnVertex(pos, idx, adj_list[idx]);
    updateVertexPositionVerlet(pos+idx, prevs+idx, accel);
    // vec3 mypos = pos[idx];
    // vec3 stretch_dir = vec3(0.0, 0.0, 0.0);
    // devarray neighbor_list = adj_list[idx];
    // for (int i=0; i < neighbor_list.nelements; i++) {
    //   unsigned int nbr_idx = neighbor_list.data[i];
    //   vec3 pos_nbr = pos[nbr_idx];
    //   auto dir_to_nbr = glm::normalize(pos_nbr - mypos);
    //   stretch_dir += dir_to_nbr * -0.1f * w;
    // }

    // calculate simple sine wave pattern
    // float freq = 4.0f;
    // float period = 10.0f; //s
    // float w = sinf(2*pi*(t + __int2float_rd(idx) / __int2float_rd(total_pts))/period) * 0.01f;

    // write output vertex
    // vec3
    // pos[idx] = vec3(mypos[0] + w, mypos[1], mypos[2]) + stretch_dir;
}

// static void APIENTRY openglCallbackFunction(
//   GLenum source,
//   GLenum type,
//   GLuint id,
//   GLenum severity,
//   GLsizei length,
//   const GLchar* message,
//   const void* userParam
// ){
//   (void)source; (void)type; (void)id;
//   (void)severity; (void)length; (void)userParam;
//   fprintf(stderr, "%s\n", message);
//   if (severity==GL_DEBUG_SEVERITY_HIGH) {
//     fprintf(stderr, "Aborting...\n");
//     abort();
//   }
// }
void freeDevArr(devarray* arr, size_t total_pts) {
  for (int i=0; i < total_pts; i++) {
    cudaFree(arr[i].data);
  }
  // cudaFree(arr);
}

devarray<unsigned int>* polysToAdjacencyList(vector<vec3> pts,
                                             vector<unsigned int> polys,
                                             devarray<unsigned int>* cuda_adj_list,
                                             devarray<float>* cuda_init_dist) {
  // Alright, this function is a bit grungy.
  // First, we make a nice, C++ typed data structure that makes sense for the
  // adjacency matrix (sparsely represented), and load the data into that:
  vector<list<unsigned int>> adjacency_list(pts.size());
  int n_edges = 0;
  for (int i=0; i < polys.size(); i += 3) {
    for (int j=0; j < 3; j++) {
      adjacency_list[polys[i + j]].push_back(polys[i + ((j+1) % 3)]);
      adjacency_list[polys[i + j]].push_back(polys[i + ((j+2) % 3)]);
    }
  }
  // We make sure to eliminate duplicate neighbors:
  for (int i=0; i < adjacency_list.size(); i++) {
    adjacency_list[i].sort();
    adjacency_list[i].unique();
    n_edges += adjacency_list[i].size();
  }

  // Now comes the grungy part. We can't really copy these C++ pointer-y data structures
  // directly to the GPU. So we use a struct that will allow us to keep track of
  // the length of a list (devarray), essentially replicating the functionality
  // of vector<>. We copy our data into a list of those, and then
  // we allocate and copy to GPU copies.
  // Because it's really a list of pointers, each pointer in the list has to be
  // allocated by CUDA and copied to the GPU.
  devarray<unsigned int>* temp_adjacency_list = new devarray<unsigned int>[adjacency_list.size()];
  devarray<float>* temp_dist_list = new devarray<float>[adjacency_list.size()];
  // For each row, copy the data, then get a GPU pointer to it.
  for (int i=0;i < adjacency_list.size(); i++) {
    auto temp_adj_data = new unsigned int[adjacency_list[i].size()];
    auto temp_dist_data = new float[adjacency_list[i].size()];
    int j = 0;
    for (auto e : adjacency_list[i]) {
      // Get the edge
      temp_adj_data[j] = e;
      // Get the initial edge length
      temp_dist_data[j] = glm::distance(pts[e], pts[i]);
      j++;
    }
    devarray<unsigned int> adj_row;
    size_t srow = sizeof(unsigned int) * adjacency_list[i].size();
    cudaMalloc((void **) &adj_row.data, srow);
    cudaMemcpy(adj_row.data, temp_adj_data, srow, cudaMemcpyHostToDevice);
    adj_row.nelements = adjacency_list[i].size();
    temp_adjacency_list[i] = adj_row;

    devarray<float> dist_row;
    srow = sizeof(float) * adjacency_list[i].size();
    cudaMalloc((void **) &dist_row.data, srow);
    cudaMemcpy(dist_row.data, temp_dist_data, srow, cudaMemcpyHostToDevice);
    dist_row.nelements = adjacency_list[i].size();
    temp_dist_list[i] = dist_row;
  }
  // Now, take the temp array (of structs containing GPU pointers)
  // we've built up, and copy it to the GPU.
  devarray* cuda_adjacency_list;
  cudaMalloc((void **) &cuda_adjacency_list, sizeof(devarray)*adjacency_list.size());
  cudaMemcpy(cuda_adjacency_list, temp_adjacency_list,
             sizeof(devarray)*adjacency_list.size(), cudaMemcpyHostToDevice);

  return cuda_adjacency_list;
}

void printAdjList(vector<list<unsigned int>> l) {
  for (auto sublist : l) {
    cout << "row: ";
    for (auto idx : sublist) {
      cout << " " << idx;
    }
    cout << endl;
  }
}


class GLManager {
 public:
   GLManager(int _winx, int _winy) : winx(_winx), winy(_winy) {}
   int init();
   void run();
   void meshLoad(std::vector<glm::vec3>, std::vector<unsigned int>);
   void runCudaVertexUpdate();

 private:
   int winx, winy;
   GLFWwindow* window;

   int n_cortex_verts; // number of vertices in the mesh
   int n_poly_idx; // number of polys in the mesh

   // OpenGL objects:
   GLuint shaderProgramID;
   GLuint mesh_buffer; // Mesh Vertex VBO
   void *cuda_mesh_vbo_buffer = NULL;
   struct cudaGraphicsResource *cuda_vbo_resource;
   GLuint poly_idx_buffer; // Polys (as index array) Index VBO
   GLuint mesh_vao; // Vertex Array for the above
   GLuint mvp_id;

   float g_fAnim = 0.0f;
   devarray* cuda_adjacency_list;

   vec3* cuda_prev_positions;
   InputHandler* input_handler;

};

int GLManager::init() {
  if( !glfwInit() )
  {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return -1;
  }
  // glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // We want OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

  this->window = glfwCreateWindow(this->winx, this->winy, "Cortex Stretcher", NULL, NULL);
  if(this->window == NULL) {
    fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(this->window);
  cout << "OpenGL version: " << glGetString(GL_VERSION);

  // Initialize CUDA
  cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

  // Initialize GLEW
  glewExperimental=true;
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return -1;
  }

  // Set GL debugging
  // GLint flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
  // if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
  // {
  //    cout << "Debug output enabled!" << endl;
  //    glEnable(GL_DEBUG_OUTPUT);
  //    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    //  glDebugMessageCallback(openglCallbackFunction, nullptr);
    //  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
  // }

  glfwSetInputMode(this->window, GLFW_STICKY_KEYS, GL_TRUE);
  glfwSetInputMode(this->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwPollEvents();
  glfwSetCursorPos(this->window, 1024/2, 768/2);

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  this->shaderProgramID = LoadShaders("vertex.glsl", "fragment.glsl");
  // Get a handle for our "MVP" uniform
	this->mvp_id = glGetUniformLocation(this->shaderProgramID, "MVP");
  this->input_handler = new InputHandler(this->window, false);
}

void GLManager::run() {
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  double lastTime = glfwGetTime();
  int nbFrames = 0;
  do {
      double currentTime = glfwGetTime();
      nbFrames++;
      if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1 sec ago
         // printf and reset timer
         printf("%f ms/frame\n", 1000.0/double(nbFrames));
         nbFrames = 0;
         lastTime += 1.0;
      }
      // for (int i=0; i < 2000; i++) {
        this->runCudaVertexUpdate();
      // }
      cudaDeviceSynchronize();
      // Swap buffers
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glUseProgram(this->shaderProgramID);

      // Compute the MVP matrix from keyboard and mouse input
  		this->input_handler->computeMatricesFromInputs();
  		glm::mat4 ProjectionMatrix = this->input_handler->getProjectionMatrix();
  		glm::mat4 ViewMatrix = this->input_handler->getViewMatrix();
  		glm::mat4 ModelMatrix = glm::mat4(1.0);
  		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
      glUniformMatrix4fv(this->mvp_id, 1, GL_FALSE, &MVP[0][0]);

      glBindVertexArray(this->mesh_vao);
      // glDrawArrays(GL_POINTS, 0, this->n_cortex_verts);
      glEnableClientState(GL_VERTEX_ARRAY);
      glDrawElements(GL_TRIANGLES, this->n_poly_idx, GL_UNSIGNED_INT, 0);
      glDisableClientState(GL_VERTEX_ARRAY);

      glfwSwapBuffers(this->window);
      glfwPollEvents();
      this->g_fAnim += 0.1f;
  // Check if the ESC key was pressed or the window was closed
  } while(glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
          glfwWindowShouldClose(this->window) == 0);
  // Cleanup VBO and shader
	glDeleteBuffers(1, &this->mesh_buffer);
  glDeleteBuffers(1, &this->poly_idx_buffer);
	glDeleteProgram(this->shaderProgramID);
	glDeleteVertexArrays(1, &this->mesh_vao);

  // Free Allocated GPU Memory
  cudaFree(this->cuda_mesh_vbo_buffer);
  // freeDevArr(this->cuda_adjacency_list, this->n_cortex_verts);
  cudaFree(this->cuda_prev_positions);

  // cudaProfilerStop();

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

void GLManager::meshLoad(std::vector<glm::vec3> pts, std::vector<unsigned int> idx) {
  // CUDA STUFF
  // cudaMalloc((void **)&d_vbo_buffer, pts.size() * sizeof(glm::vec3));
  this->cuda_adjacency_list = polysToAdjacencyList(pts, idx);
  cudaMalloc((void **) &this->cuda_prev_positions, sizeof(vec3)*pts.size());
  cudaMemcpy(this->cuda_prev_positions, &pts[0], sizeof(vec3)*pts.size(),
             cudaMemcpyHostToDevice);

	glGenBuffers(1, &this->mesh_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, this->mesh_buffer);
	glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(glm::vec3),
               &pts[0], GL_DYNAMIC_DRAW);

 	glGenBuffers(1, &this->poly_idx_buffer);
 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->poly_idx_buffer);
 	glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned int),
               &idx[0] , GL_STATIC_DRAW);

  glGenVertexArrays(1, &this->mesh_vao);
  glBindVertexArray(this->mesh_vao);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, this->mesh_buffer);
  // CUDA STUFF
  cudaGraphicsGLRegisterBuffer(&this->cuda_vbo_resource, this->mesh_buffer,
                               cudaGraphicsMapFlagsWriteDiscard);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->poly_idx_buffer);

  this->n_poly_idx = idx.size();
  this->n_cortex_verts = pts.size();
}

void GLManager::runCudaVertexUpdate() {
  cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0);
  size_t num_bytes;
  vec3 *dptr;
  cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                        this->cuda_vbo_resource);
  int num_pts = this-> n_cortex_verts;
  int num_sms = 28;
  int pts_per_block = num_pts / num_sms + 1;
  int num_blocks = 28;
  if (pts_per_block > 2048) {
    pts_per_block = 2048;
    num_blocks = num_pts / 2048 + 1;
  }
  int t = time(0);
  simple_distortion<<< num_blocks, pts_per_block >>>(dptr, num_pts, this->g_fAnim,
                                                 this->cuda_adjacency_list,
                                                 this->cuda_prev_positions);
}

std::vector<glm::vec3> giiToVertices(giiDataArray *d) {
  std::vector<glm::vec3> out_vertices;
  int c, size;
  float *newarr = new float[d->nvals];
  assert(d->datatype == NIFTI_TYPE_FLOAT32);
  gifti_copy_data_as_float(newarr, NIFTI_TYPE_FLOAT32, d->data, d->datatype, d->nvals);

  float maxx, maxy, minx, miny = 0;
  for (int i = 0; i < d->nvals; i += 3) {
    // printf("Vertex read: %.3f, %.3f\n", newarr[i], newarr[i+1]);
    if (newarr[i] > maxx) {
      maxx = newarr[i];
    } else if (newarr[i] < minx) {
      minx = newarr[i];
    }
    if (newarr[i+1] > maxy) {
      maxy = newarr[i+1];
    } else if (newarr[i+1] < miny) {
      miny = newarr[i+1];
    }
    out_vertices.push_back(glm::vec3(newarr[i], newarr[i+1], 0.0));
  }

  // Normalize the vector:
  for (int i = 0; i < out_vertices.size(); i++) {
    out_vertices[i] = glm::vec3(out_vertices[i].x / (maxx - minx),
                                out_vertices[i].y / (maxy - miny), 0.0);
  }
  return out_vertices;
}

std::vector<unsigned int> giiToIndices(giiDataArray *d) {
  std::vector<unsigned int> out_vertices;
  int c, size;
  assert(d->datatype == NIFTI_TYPE_INT32);
  float *newarr = new float[d->nvals];
  gifti_copy_data_as_float(newarr, NIFTI_TYPE_FLOAT32, d->data, d->datatype, d->nvals);
  cout << "reading indices" << endl;
  for (int i = 0; i < d->nvals; i++) { //d->nvals
    out_vertices.push_back(static_cast<unsigned int>(newarr[i]));
  }
  cout << "idx read" << endl;
  return out_vertices;
}

int main(int argc, char *argv[]) {
  gifti_image *out_im;
  cout << "input argument: " << argv[1] << endl;
  gifti_disp_lib_version();
  out_im = gifti_read_image(argv[1], 1);
  cout << "valid image: " << gifti_valid_gifti_image(out_im, 1) << endl;
  gifti_disp_LabelTable("Label Table:", &out_im->labeltable);
  for (int i = 0; i < out_im->numDA; i++) {
    gifti_disp_DataArray("datarray: ", out_im->darray[i], 1);
  }


  giiDataArray *pts = out_im->darray[0];
  giiDataArray *triangles = out_im->darray[1];
  gifti_disp_raw_data(triangles->data, triangles->datatype, 100, 1, stdout);

  cout << "successfully read data" << endl;

  GLManager* manager = new GLManager(1024, 768);
  manager->init();
  // glEnable(GL_DEBUG_OUTPUT);
  auto adj_list = polysToAdjacencyList(giiToVertices(pts), giiToIndices(triangles));
  // printAdjList(adj_list);
  manager->meshLoad(giiToVertices(pts), giiToIndices(triangles));

  cout << "loaded buffers " << endl;
  manager->run();

  return 0;
}
