#ifndef CONTROLS_H
#define CONTROLS_H

class InputHandler {
 public:
	InputHandler(GLFWwindow* _window);
	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();
	void computeMatricesFromInputs();
  void updateFoV(double offset);

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
};

#endif
