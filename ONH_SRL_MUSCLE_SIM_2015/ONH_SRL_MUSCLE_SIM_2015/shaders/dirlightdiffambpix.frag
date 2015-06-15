#version 330

uniform	sampler2D texUnit;
uniform mat4 material;
uniform float texCount;
uniform float shininess;
uniform float opacity;

in vec3 Normal;
in vec2 TexCoord;

out vec4 output;

void main()
{
	vec4 color;
	vec4 amb;
	float intensity;
	float intensityDown;
	vec3 lightDir;
	vec3 lightDir2;
	vec3 n;
	vec4 diffuse = material[0];
	vec4 ambient = material[1];
	vec4 specular = material[2];
	vec4 emissive = material[3];
	
	lightDir = normalize(vec3(10.0,30.0,10.0));
	lightDir2 = normalize(vec3(-10.0,-30.0,50.0));

	n = normalize(Normal);
	intensity = max(dot(lightDir,n),0.0);
	intensityDown = max(dot(lightDir2,n),0.0);

	diffuse.a = opacity;

	if (texCount > 0.1)
	{
		color = diffuse * texture(texUnit, TexCoord);
		amb = color * 0.33;
	}
	else
	{
		color = diffuse;
		amb = ambient;
		amb.a = opacity;
	}
	//output = (color * max(intensity,intensityDown)) + amb;
	output = (color * (intensity+intensityDown)) + amb;
}
