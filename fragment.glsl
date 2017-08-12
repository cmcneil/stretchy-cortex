#version 400

// Interpolated values from the vertex shaders
in vec3 vp;

// Ouput data
out vec4 frag_color;

void main(){

	// Output color = color of the texture at the specified UV
	frag_color = vec4(0.0, 1.0, 0.0, 1.0);
}
