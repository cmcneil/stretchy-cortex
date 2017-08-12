extern "C" {
  #include <gifti_io.h>
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <glfw3.h>
#include <glm/glm.hpp>

#include "shader.h"
#include "controls.h"

using namespace glm;
using namespace std;

static void APIENTRY openglCallbackFunction(
  GLenum source,
  GLenum type,
  GLuint id,
  GLenum severity,
  GLsizei length,
  const GLchar* message,
  const void* userParam
){
  (void)source; (void)type; (void)id;
  (void)severity; (void)length; (void)userParam;
  fprintf(stderr, "%s\n", message);
  if (severity==GL_DEBUG_SEVERITY_HIGH) {
    fprintf(stderr, "Aborting...\n");
    abort();
  }
}


class GLManager {
 public:
   GLManager(int _winx, int _winy) : winx(_winx), winy(_winy) {}
   int init();
   void run();
   void pointsToVBO(std::vector<glm::vec3>);

 private:
   int winx, winy;
   GLFWwindow* window;
   GLuint shaderProgramID;
   int n_cortex_verts;
   GLuint mesh_buffer;
   GLuint triangle_vao;
   GLuint mesh_vao;
   GLuint mvp_id;
   InputHandler* input_handler;
};

int GLManager::init() {
  if( !glfwInit() )
  {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return -1;
  }
  glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
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

  GLint flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
  if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
  {
     cout << "Debug output enabled!" << endl;
     glEnable(GL_DEBUG_OUTPUT);
     glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    //  glDebugMessageCallback(openglCallbackFunction, nullptr);
    //  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
  }

  // Initialize GLEW
  glewExperimental=true;
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return -1;
  }
  glfwSetInputMode(this->window, GLFW_STICKY_KEYS, GL_TRUE);
  glfwSetInputMode(this->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwPollEvents();
  glfwSetCursorPos(this->window, 1024/2, 768/2);

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  this->shaderProgramID = LoadShaders("vertex.glsl", "fragment.glsl");
  // Get a handle for our "MVP" uniform
	this->mvp_id = glGetUniformLocation(this->shaderProgramID, "MVP");
  this->input_handler = new InputHandler(this->window);
}

void GLManager::run() {
  glfwSetCursorPos(this->window, 1024/2, 768/2);
  glfwSetCursorPos(this->window, 1024/2, 768/2+1);
  glfwSetCursorPos(this->window, 1024/2, 768/2);
  do {
      // Draw nothing, see you in tutorial 2 !
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

      // glBindVertexArray(this->triangle_vao);
      // glDrawArrays(GL_TRIANGLES, 0, 3);
      glBindVertexArray(this->mesh_vao);
      glDrawArrays(GL_POINTS, 0, this->n_cortex_verts);

      glfwSwapBuffers(this->window);
      glfwPollEvents();
  // Check if the ESC key was pressed or the window was closed
  } while(glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
          glfwWindowShouldClose(this->window) == 0);
  // Cleanup VBO and shader
	glDeleteBuffers(1, &this->mesh_buffer);
	glDeleteProgram(this->shaderProgramID);
	glDeleteVertexArrays(1, &this->mesh_vao);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

void GLManager::pointsToVBO(std::vector<glm::vec3> points) {
  glGenBuffers(1, &this->mesh_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, this->mesh_buffer);
  glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(glm::vec3),
               &points[0], GL_STATIC_DRAW);
  glGenVertexArrays(1, &this->mesh_vao);
  glBindVertexArray(this->mesh_vao);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, this->mesh_buffer);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  this->n_cortex_verts = points.size();
}

std::vector<glm::vec3> giiToVertices(giiDataArray *d) {
  std::vector<glm::vec3> out_vertices;
  int c, size;
  float *newarr = new float[d->nvals];
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


  giiDataArray *d = out_im->darray[0];
  gifti_disp_raw_data(d->data, d->datatype, 5, 1, stdout);

  cout << "successfully read data" << endl;

  GLManager* manager = new GLManager(1024, 768);
  manager->init();
  glEnable(GL_DEBUG_OUTPUT);
  manager->pointsToVBO(giiToVertices(d));
  manager->run();

  return 0;
}
