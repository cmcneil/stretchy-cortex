#version 400

in vec3 vp;

// Model view projection matrix:
uniform mat4 MVP;

void main(){

	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vp, 1);
};
