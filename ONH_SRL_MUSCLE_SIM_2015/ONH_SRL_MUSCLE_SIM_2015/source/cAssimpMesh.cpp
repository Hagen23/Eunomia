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
	//material.material[0] = material.diffuse[0];
	//material.material[4] = material.diffuse[1];
	//material.material[8] = material.diffuse[2];
	//material.material[12] = material.diffuse[3];

	//material.material[1] = material.ambient[0];
	//material.material[5] = material.ambient[1];
	//material.material[9] = material.ambient[2];
	//material.material[13] = material.ambient[3];

	//material.material[2] = material.specular[0];
	//material.material[6] = material.specular[1];
	//material.material[10] = material.specular[2];
	//material.material[14] = material.specular[3];

	//material.material[3] = material.emissive[0];
	//material.material[7] = material.emissive[1];
	//material.material[11] = material.emissive[2];
	//material.material[15] = material.emissive[3];

	for (unsigned int i = 0; i < 4; i++)
	{
		material.material[0+i] = material.diffuse[i];
		material.material[4+i] = material.ambient[i];
		material.material[8+i] = material.specular[i];
		material.material[12+i] = material.emissive[i];
	}
}
