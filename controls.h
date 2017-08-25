#ifndef CONTROLS_H
#define CONTROLS_H

class InputHandler {
 public:
	InputHandler(GLFWwindow* _window, bool _enable_rotate);
	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();
	void computeMatricesFromInputs();
  void updateFoV(double offset);
  float getForceData();
  float getForceMesh();

 private:
	GLFWwindow* window;
	glm::mat4 ViewMatrix;
 	glm::mat4 ProjectionMatrix;
	// Initial position : on +Z
	glm::vec3 position = glm::vec3( 0, 0, 1 );
	// Initial horizontal angle : toward -Z
	float horizontalAngle = 3.14f;
	// Initial vertical angle : none
	float verticalAngle = 0.0f;
	// Initial Field of View
	float initialFoV = 80.0f;
  float FoV = initialFoV;
	float speed = 3.0f; // 3 units / second
	float mouseSpeed = 0.005f;
  bool enable_rotate;

  float k_data = 0.00001f;
  float k_mesh = 0.01f;
};

#endif
