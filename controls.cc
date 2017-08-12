// Include GLFW
#include <glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <functional>
using namespace glm;

#include "controls.h"

// ### CALLBACK FUNCTIONS ### //
// Callback function to zoom the FOV when sent the callback.
void handleScrollZoom(GLFWwindow* win, double xoffset, double yoffset) {
	void *usrptr = glfwGetWindowUserPointer(win);
	InputHandler* handler = static_cast<InputHandler*>(usrptr);
	handler->updateFoV(5*yoffset);
}

// ### CLASS METHODS ### //
InputHandler::InputHandler(GLFWwindow* window) {
	this->window = window;

	glfwSetWindowUserPointer(this->window, this);
	glfwSetScrollCallback(this->window, handleScrollZoom);
}

void InputHandler::updateFoV(double offset) {
	this->FoV = this->FoV - offset;
}

glm::mat4 InputHandler::getViewMatrix(){
	return ViewMatrix;
}
glm::mat4 InputHandler::getProjectionMatrix(){
	return ProjectionMatrix;
}

void InputHandler::computeMatricesFromInputs(){

	// glfwGetTime is called only once, the first time this function is called
	static double lastTime = glfwGetTime();

	// Compute time difference between current and last frame
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);

	// Get mouse position
	double xpos, ypos;
	glfwGetCursorPos(this->window, &xpos, &ypos);

	// Reset mouse position for next frame
	glfwSetCursorPos(this->window, 1024/2, 768/2);

	// Compute new orientation
	this->horizontalAngle += this->mouseSpeed * float(1024/2 - xpos );
	this->verticalAngle   += this->mouseSpeed * float( 768/2 - ypos );

	// Direction : Spherical coordinates to Cartesian coordinates conversion
	glm::vec3 direction(
		cos(this->verticalAngle) * sin(this->horizontalAngle),
		sin(this->verticalAngle),
		cos(this->verticalAngle) * cos(this->horizontalAngle)
	);

	// Right vector
	glm::vec3 right = glm::vec3(
		sin(this->horizontalAngle - 3.14f/2.0f),
		0,
		cos(this->horizontalAngle - 3.14f/2.0f)
	);

	// Up vector
	glm::vec3 up = glm::cross( right, direction );

	// Move forward
	if (glfwGetKey( window, GLFW_KEY_UP ) == GLFW_PRESS){
		position += direction * deltaTime * this->speed;
	}
	// Move backward
	if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS){
		position -= direction * deltaTime * this->speed;
	}
	// Strafe right
	if (glfwGetKey( window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
		position += right * deltaTime * this->speed;
	}
	// Strafe left
	if (glfwGetKey( window, GLFW_KEY_LEFT ) == GLFW_PRESS){
		position -= right * deltaTime * this->speed;
	}

	// Projection matrix : 45� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	this->ProjectionMatrix = glm::perspective(glm::radians(this->FoV), 4.0f / 3.0f, 0.1f, 100.0f);
	// Camera matrix
	this->ViewMatrix       = glm::lookAt(
								position,           // Camera is here
								position+direction, // and looks here : at the same position, plus "direction"
								up                  // Head is up (set to 0,-1,0 to look upside-down)
						   );

	// For the next frame, the "last time" will be "now"
	lastTime = currentTime;
}
