#version 330

uniform mat4 projMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

out vec4 vertexPos;
out vec2 TexCoord;
out vec3 Normal;

void main()
{
	Normal = normalize(vec3(viewMatrix * modelMatrix * vec4(normal,0.0)));	
	TexCoord = vec2(texCoord);
	//vertexPos = projMatrix * viewMatrix * modelMatrix * vec4(position,1.0);
	vertexPos = viewMatrix * modelMatrix * vec4(position,1.0);
	vertexPos.w = 1.0;
	gl_Position = vertexPos;
}
