#version 330

uniform	sampler2D texUnit;
uniform mat4 material;
uniform float texCount;
uniform float shininess;

in vec4 vertexPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 output;

void main()
{
	vec4 color;
	vec4 amb;
	float intensity;
	vec3 lightDir;
	vec3 n;
	vec4 diffuse = vec4(material[0][0],material[0][1],material[0][2],material[0][3]);
	vec4 ambient = material[1];
	vec4 specular = material[2];
	vec4 emissive = material[3];
	
	lightDir = normalize(vec3(1.0,1.0,1.0));
	n = normalize(Normal);	
	intensity = max(dot(lightDir,n),0.0);
	
	if (texCount > 0.1)
	{
		color = texture(texUnit, TexCoord);
		amb = color * 0.33;
	}
	else
	{
		color = diffuse;
		amb = ambient;
	}
	output = (color * intensity) + amb;
}
