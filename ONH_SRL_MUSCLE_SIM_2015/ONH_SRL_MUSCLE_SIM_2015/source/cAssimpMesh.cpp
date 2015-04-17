#include "cAssimpMesh.h"


AssimpMesh::AssimpMesh()
{
	numFaces = 0;
	texIndex = 0;
	mVAO = new QOpenGLVertexArrayObject();
	for (unsigned int i = 0; i < 4; i++)
	{
		material.diffuse[i] = 0.0f;
		material.ambient[i] = 0.0f;
		material.specular[i] = 0.0f;
		material.emissive[i] = 0.0f;
	}
	for (unsigned int i = 0; i < 16; i++)
	{
		material.material[i] = 0.0f;
	}
	material.shininess = 50.0f;
	material.texCount = 0;
}

AssimpMesh::~AssimpMesh()
{
}

void AssimpMesh::pack(void)
{
	for (unsigned int i = 0; i < 4; i++)
	{
		material.material[0+i] = material.diffuse[i];
		material.material[4+i] = material.ambient[i];
		material.material[8+i] = material.specular[i];
		material.material[12+i] = material.emissive[i];
	}
}
