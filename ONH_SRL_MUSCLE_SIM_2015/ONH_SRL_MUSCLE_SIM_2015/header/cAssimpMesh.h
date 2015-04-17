#pragma once

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QObject>
#include <QOpenGLContext>

#include "cMacros.h"

#ifndef __ASSIMP_MESH
#define __ASSIMP_MESH

class AssimpMesh
{
public:
	AssimpMesh();
	~AssimpMesh();

	void pack(void);

	QOpenGLVertexArrayObject* mVAO;
	QOpenGLBuffer mVertexPositionBuffer;
	QOpenGLBuffer mVertexNormalBuffer;
	QOpenGLBuffer mVertexTextureBuffer;
	QOpenGLBuffer mFaceIndicesBuffer;
	int numFaces;
	GLuint texIndex;
	ShaderMaterial material;
};

#endif /*__ASSIMP_MESH*/