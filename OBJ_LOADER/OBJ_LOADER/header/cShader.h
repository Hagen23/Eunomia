#pragma once
#include "glew.h"

#include <string>
#include <iostream>

#include "cMacros.h"
#include "cTextIO.h"

using namespace std;

#ifndef __SHADER
#define __SHADER

class Shader
{
public:
	Shader( string _vfn, string ffn );
	~Shader();

	void	compile(void);

	GLuint	getProgram(void);
	GLuint	getVertexLoc(void);
	GLuint	getNormalLoc(void);
	GLuint	getTexCoordLoc(void);
	GLuint	getMatricesUniLoc(void);
	GLuint	getMaterialUniLoc(void);
	GLuint	getTexUnit(void);

private:
	void	printShaderInfoLog(GLuint obj);
	void	printProgramInfoLog(GLuint obj);
	void	setupShader(void);

	// Read text files:
	TextIO*	textio;
	// Program and Shader Identifiers:
	GLuint	program;
	GLuint	vertexShader;
	GLuint	fragmentShader;
	// Vertex Attribute Locations:
	GLuint	vertexLoc;
	GLuint	normalLoc;
	GLuint	texCoordLoc;
	// Uniform Bindings Points:
	GLuint	matricesUniLoc;
	GLuint	materialUniLoc;
	// The sampler uniform for textured models
	// we are assuming a single texture so this will
	// always be texture unit 0:
	GLuint	texUnit;

	string	vertexFileName;
	string	fragmentFileName;
};

#endif
