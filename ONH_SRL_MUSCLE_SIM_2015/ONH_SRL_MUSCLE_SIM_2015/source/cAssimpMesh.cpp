#include "cAssimpMesh.h"


AssimpMesh::AssimpMesh()
{
	numFaces = 0;
	texIndex = 0;
	mVAO = new QOpenGLVertexArrayObject();
	material.material.fill(0.0f);
	material.shininess = 50.0f;
	material.texCount = 0;
	material.opacity = 1.0f;
}

AssimpMesh::~AssimpMesh()
{
}

void AssimpMesh::pack(void)
{
	material.material = QMatrix4x4(
		material.diffuse[0], material.diffuse[1], material.diffuse[2], material.diffuse[3],
		material.ambient[0], material.ambient[1], material.ambient[2], material.ambient[3],
		material.specular[0], material.specular[1], material.specular[2], material.specular[3],
		material.emissive[0], material.emissive[1], material.emissive[2], material.emissive[3]
		);
	material.material = material.material.transposed();
}
