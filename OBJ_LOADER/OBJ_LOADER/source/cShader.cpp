#include "cShader.h"

Shader::Shader( string vfn, string ffn )
{
	program				= 0;
	vertexShader		= 0;
	fragmentShader		= 0;

	vertexLoc			= 0;
	normalLoc			= 1;
	texCoordLoc			= 2;

	matricesUniLoc		= 1;
	materialUniLoc		= 2;

	// The sampler uniform for textured models
	// we are assuming a single texture so this will
	//always be texture unit 0
	texUnit				= 0;

	vertexFileName		= vfn;
	fragmentFileName	= ffn;

	textio				= new TextIO();
}

Shader::~Shader()
{
	delete textio;
}

void Shader::compile(void)
{
	setupShader();
}

void Shader::printShaderInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		cout << infoLog << endl;
		free(infoLog);
	}
}

void Shader::printProgramInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
		cout << infoLog << endl;
		free(infoLog);
	}
}

void Shader::setupShader(void)
{
	char *vs = NULL, *fs = NULL, *fs2 = NULL;

	GLuint p, v, f;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	string vss = textio->textFileRead(string(vertexFileName));
	string fss = textio->textFileRead(string(fragmentFileName));

	const char * vv = vss.c_str();
	const char * ff = fss.c_str();

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs);
	free(fs);

	glCompileShader(v);
	glCompileShader(f);

	printShaderInfoLog(v);
	printShaderInfoLog(f);

	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	glBindFragDataLocation(p, 0, "output");

	glBindAttribLocation(p, vertexLoc, "position");
	glBindAttribLocation(p, normalLoc, "normal");
	glBindAttribLocation(p, texCoordLoc, "texCoord");

	glLinkProgram(p);
	glValidateProgram(p);
	printProgramInfoLog(p);

	program = p;
	vertexShader = v;
	fragmentShader = f;

	GLuint k = glGetUniformBlockIndex(p, "Matrices");
	glUniformBlockBinding(p, k, matricesUniLoc);
	glUniformBlockBinding(p, glGetUniformBlockIndex(p, "Material"), materialUniLoc);

	texUnit = glGetUniformLocation(p, "texUnit");
}

GLuint	Shader::getProgram(void)
{
	return program;
}

GLuint	Shader::getVertexLoc(void)
{
	return vertexLoc;
}

GLuint	Shader::getNormalLoc(void)
{
	return normalLoc;
}

GLuint	Shader::getTexCoordLoc(void)
{
	return texCoordLoc;
}

GLuint	Shader::getMatricesUniLoc(void)
{
	return matricesUniLoc;
}

GLuint	Shader::getMaterialUniLoc(void)
{
	return materialUniLoc;
}

GLuint	Shader::getTexUnit(void)
{
	return texUnit;
}
