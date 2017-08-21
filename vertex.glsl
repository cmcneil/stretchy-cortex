#version 450 core

// in vec3 vp;
layout (location = 0) in vec3 vp;
// Model view projection matrix:
uniform mat4 MVP;

void main(){

	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vp.x, vp.y, vp.z, 1.0);
};
