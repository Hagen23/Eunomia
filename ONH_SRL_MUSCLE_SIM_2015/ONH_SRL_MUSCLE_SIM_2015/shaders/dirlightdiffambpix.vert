#version 330

uniform mat4 projMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

in vec3 position;
in vec3 normal;
in vec2 texCoord;

out vec3 Normal;
out vec2 TexCoord;

void main()
{
	vec4 vertexPos;
	Normal = normalize(vec3(viewMatrix * modelMatrix * vec4(normal,0.0)));	
	TexCoord = vec2(texCoord);
	vertexPos = projMatrix * (viewMatrix * modelMatrix) * vec4(position,1.0);
	gl_Position = vertexPos;
}
