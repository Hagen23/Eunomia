#include "QtOpenGLScene.h"
#include "glassert.h"

QtOpenGLScene::QtOpenGLScene() : mShaderProgram()
{

}

int QtOpenGLScene::initialize()
{
	//modelname = "assets/bench.obj";
	//modelname = "assets/sphere.obj";
	//modelname = "assets/sphere_notex.obj";
	//modelname = "assets/spider.obj";
	//modelname = "assets/testmixed.obj";
	//modelname = "assets/WusonOBJ.obj";
	modelname = "assets/arm.obj";

	string basepath = StringUtils::getBasePath(modelname);
	tex_manager = new TextureManager(basepath);
	assimp_manager = new AssimpManager(modelname);

	if (!assimp_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(assimp_manager->getScene()))
	{
		return 4;
	}
	prepareShaderProgram();
	genVAOsAndUniformBuffer(assimp_manager->getScene());
	mShaderProgram.bind();
	texUnit = mShaderProgram.uniformLocation("texUnit");
	return 0;
}

void QtOpenGLScene::update(float t)
{
    Q_UNUSED(t);
}

void QtOpenGLScene::render(int rotX, int rotY, int rotZ)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	renderScene(rotX, rotY, rotZ);
}

void QtOpenGLScene::resize(int width, int height)
{
	float ratio;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (height == 0) height = 1;
	// Set the viewport to be the entire window
	glAssert(glViewport(0, 0, width, height));
	ratio = (1.0f * width) / height;
	buildProjectionMatrix(53.13f, ratio, 0.1f, 100.0f);
	glCheckError();
	qDebug() << "RESIZED TO " << width << "X" << height << "\n";
}

void QtOpenGLScene::prepareShaderProgram()
{
	if (!mShaderProgram.addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/dirlightdiffambpix.vert"))
    {
        qCritical() << "error";
    }
	if (!mShaderProgram.addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/dirlightdiffambpix.frag"))
    {
        qCritical() << "error";
    }
    if (!mShaderProgram.link())
    {
        qCritical() << "error";
    }

    glCheckError();
}

void QtOpenGLScene::set_float4(float f[4], float a, float b, float c, float d)
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

void QtOpenGLScene::color4_to_float4(const aiColor4D *c, float f[4])
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}

void QtOpenGLScene::genVAOsAndUniformBuffer(const aiScene* sc)
{
	// For each mesh
	for (unsigned int n = 0; n < sc->mNumMeshes; ++n)
	{
		AssimpMesh	aMesh;
		aMesh.mVertexPositionBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		aMesh.mVertexNormalBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		aMesh.mVertexTextureBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
		aMesh.mFaceIndicesBuffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
		const aiMesh* mesh = sc->mMeshes[n];

		// create array with faces
		// have to convert from Assimp format to array
		unsigned int *faceArray;
		faceArray = (unsigned int *)malloc(sizeof(unsigned int) * mesh->mNumFaces * 3);
		unsigned int faceIndex = 0;

		for (unsigned int t = 0; t < mesh->mNumFaces; ++t)
		{
			const aiFace* face = &mesh->mFaces[t];

			memcpy(&faceArray[faceIndex], face->mIndices, 3 * sizeof(unsigned int));
			faceIndex += 3;
		}
		aMesh.numFaces = sc->mMeshes[n]->mNumFaces;

		// generate Vertex Array for mesh
		aMesh.mVAO->create();
		aMesh.mVAO->bind();
		// buffer for faces
		aMesh.mFaceIndicesBuffer.create();
		aMesh.mFaceIndicesBuffer.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
		aMesh.mFaceIndicesBuffer.bind();
		aMesh.mFaceIndicesBuffer.allocate(faceArray, aMesh.numFaces * 3 * sizeof(unsigned int));
		// buffer for vertex positions
		if (mesh->HasPositions())
		{
			aMesh.mVertexPositionBuffer.create();
			aMesh.mVertexPositionBuffer.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
			aMesh.mVertexPositionBuffer.bind();
			aMesh.mVertexPositionBuffer.allocate(mesh->mVertices, mesh->mNumVertices * 3 * sizeof(float));

			mShaderProgram.bind();
			mShaderProgram.enableAttributeArray("position");
			mShaderProgram.setAttributeBuffer("position", GL_FLOAT, 0, 3);
		}

		// buffer for vertex normals
		if (mesh->HasNormals())
		{
			aMesh.mVertexNormalBuffer.create();
			aMesh.mVertexNormalBuffer.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
			aMesh.mVertexNormalBuffer.bind();
			aMesh.mVertexNormalBuffer.allocate(mesh->mNormals, mesh->mNumVertices * 3 * sizeof(float));

			mShaderProgram.bind();
			mShaderProgram.enableAttributeArray("normal");
			mShaderProgram.setAttributeBuffer("normal", GL_FLOAT, 0, 3);
		}

		// buffer for vertex texture coordinates
		if (mesh->HasTextureCoords(0))
		{
			float *texCoords = (float *)malloc(sizeof(float) * 2 * mesh->mNumVertices);
			for (unsigned int k = 0; k < mesh->mNumVertices; ++k)
			{
				texCoords[k * 2] = mesh->mTextureCoords[0][k].x;
				texCoords[k * 2 + 1] = mesh->mTextureCoords[0][k].y;
			}

			aMesh.mVertexTextureBuffer.create();
			aMesh.mVertexTextureBuffer.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
			aMesh.mVertexTextureBuffer.bind();
			aMesh.mVertexTextureBuffer.allocate(texCoords, mesh->mNumVertices * 2 * sizeof(float));

			mShaderProgram.bind();
			mShaderProgram.enableAttributeArray("texCoord");
			mShaderProgram.setAttributeBuffer("texCoord", GL_FLOAT, 0, 2);
		}

		// create material uniform buffer
		aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];

		aiString texPath;	//contains filename of texture
		if (AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath))
		{
			// Bind texture:
			unsigned int texId = tex_manager->textureIdMap[texPath.data];
			aMesh.texIndex = texId;
			aMesh.material.texCount = 1;
		}
		else
			aMesh.material.texCount = 0;

		float c[4];
		set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
		aiColor4D diffuse;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
			color4_to_float4(&diffuse, c);

		//if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_TRANSPARENT), &)
		//	memcpy(aMat.diffuse, c, sizeof(c));

		set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
		aiColor4D ambient;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
			color4_to_float4(&ambient, c);
		memcpy(aMesh.material.ambient, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D specular;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
			color4_to_float4(&specular, c);
		memcpy(aMesh.material.specular, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D emission;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
			color4_to_float4(&emission, c);
		memcpy(aMesh.material.emissive, c, sizeof(c));

		float shininess = 0.0;
		unsigned int max;
		aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
		aMesh.material.shininess = shininess;

		aMesh.pack();
		myMeshes.push_back(aMesh);
	}
	glCheckError();
}

void QtOpenGLScene::crossProduct(float* a, float* b, float* res)
{
	res[0] = a[1] * b[2] - b[1] * a[2];
	res[1] = a[2] * b[0] - b[2] * a[0];
	res[2] = a[0] * b[1] - b[0] * a[1];
}

void QtOpenGLScene::normalize(float* a)
{
	float mag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	a[0] /= mag;
	a[1] /= mag;
	a[2] /= mag;
}

void QtOpenGLScene::pushMatrix()
{
	float *aux = (float *)malloc(sizeof(float) * 16);
	memcpy(aux, modelMatrix, sizeof(float) * 16);
	matrixStack.push_back(aux);
}

void QtOpenGLScene::popMatrix()
{
	float *m = matrixStack[matrixStack.size() - 1];
	memcpy(modelMatrix, m, sizeof(float) * 16);
	matrixStack.pop_back();
	free(m);
}

void QtOpenGLScene::setIdentityMatrix(float* mat, int size)
{
	// fill matrix with 0s
	for (int i = 0; i < size * size; ++i)
		mat[i] = 0.0f;
	// fill diagonal with 1s
	for (int i = 0; i < size; ++i)
		mat[i + i * size] = 1.0f;
}

void QtOpenGLScene::multMatrix(float* a, float* b)
{
	float res[16];
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			res[j * 4 + i] = 0.0f;
			for (int k = 0; k < 4; ++k)
			{
				res[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
			}
		}
	}
	memcpy(a, res, 16 * sizeof(float));
}

void QtOpenGLScene::setTranslationMatrix(float* mat, float x, float y, float z)
{
	setIdentityMatrix(mat, 4);
	mat[12] = x;
	mat[13] = y;
	mat[14] = z;
}

void QtOpenGLScene::setScaleMatrix(float* mat, float sx, float sy, float sz)
{
	setIdentityMatrix(mat, 4);
	mat[0] = sx;
	mat[5] = sy;
	mat[10] = sz;
}

void QtOpenGLScene::setRotationMatrix(float* mat, float angle, float x, float y, float z)
{
	float radAngle = (float)DEG2RAD * angle;
	float co = cos(radAngle);
	float si = sin(radAngle);
	float x2 = x*x;
	float y2 = y*y;
	float z2 = z*z;

	mat[0] = x2 + (y2 + z2) * co;
	mat[4] = x * y * (1 - co) - z * si;
	mat[8] = x * z * (1 - co) + y * si;
	mat[12] = 0.0f;

	mat[1] = x * y * (1 - co) + z * si;
	mat[5] = y2 + (x2 + z2) * co;
	mat[9] = y * z * (1 - co) - x * si;
	mat[13] = 0.0f;

	mat[2] = x * z * (1 - co) - y * si;
	mat[6] = y * z * (1 - co) + x * si;
	mat[10] = z2 + (x2 + y2) * co;
	mat[14] = 0.0f;

	mat[3] = 0.0f;
	mat[7] = 0.0f;
	mat[11] = 0.0f;
	mat[15] = 1.0f;
}

void QtOpenGLScene::setModelMatrix()
{
	mShaderProgram.bind();
	mShaderProgram.setUniformValue("modelMatrix", QMatrix4x4(modelMatrix));
}

void QtOpenGLScene::translate(float x, float y, float z)
{
	float aux[16];
	setTranslationMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

void QtOpenGLScene::rotate(float angle, float x, float y, float z)
{
	float aux[16];
	setRotationMatrix(aux, angle, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

void QtOpenGLScene::scale(float x, float y, float z)
{
	float aux[16];
	setScaleMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

void QtOpenGLScene::buildProjectionMatrix(float fov, float ratio, float nearp, float farp)
{
	float f = 1.0f / tanf(fov * ((float)M_PI / 360.0f));
	setIdentityMatrix(projMatrix, 4);

	projMatrix[0] = f / ratio;
	projMatrix[1 * 4 + 1] = f;
	projMatrix[2 * 4 + 2] = (farp + nearp) / (nearp - farp);
	projMatrix[3 * 4 + 2] = (2.0f * farp * nearp) / (nearp - farp);
	projMatrix[2 * 4 + 3] = -1.0f;
	projMatrix[3 * 4 + 3] = 0.0f;

	mShaderProgram.bind();
	mShaderProgram.setUniformValue("projMatrix", QMatrix4x4(projMatrix));
	glCheckError();
}

void QtOpenGLScene::setCamera(float posX, float posY, float posZ, float lookAtX, float lookAtY, float lookAtZ)
{
	float dir[3], right[3], up[3];
	up[0] = 0.0f;	up[1] = 1.0f;	up[2] = 0.0f;

	dir[0] = (lookAtX - posX);
	dir[1] = (lookAtY - posY);
	dir[2] = (lookAtZ - posZ);
	normalize(dir);

	crossProduct(dir, up, right);
	normalize(right);

	crossProduct(right, dir, up);
	normalize(up);

	float aux[16];

	viewMatrix[0] = right[0];
	viewMatrix[4] = right[1];
	viewMatrix[8] = right[2];
	viewMatrix[12] = 0.0f;

	viewMatrix[1] = up[0];
	viewMatrix[5] = up[1];
	viewMatrix[9] = up[2];
	viewMatrix[13] = 0.0f;

	viewMatrix[2] = -dir[0];
	viewMatrix[6] = -dir[1];
	viewMatrix[10] = -dir[2];
	viewMatrix[14] = 0.0f;

	viewMatrix[3] = 0.0f;
	viewMatrix[7] = 0.0f;
	viewMatrix[11] = 0.0f;
	viewMatrix[15] = 1.0f;

	setTranslationMatrix(aux, -posX, -posY, -posZ);

	multMatrix(viewMatrix, aux);

	mShaderProgram.bind();
	mShaderProgram.setUniformValue("viewMatrix", QMatrix4x4(viewMatrix));
}

void QtOpenGLScene::recursive_render(const aiScene *sc, const aiNode* nd)
{
	// Get node transformation matrix
	aiMatrix4x4 m = nd->mTransformation;
	// OpenGL matrices are column major
	m.Transpose();
	// save model matrix and apply node transformation
	pushMatrix();

	float aux[16];
	memcpy(aux, &m, sizeof(float) * 16);
	multMatrix(modelMatrix, aux);
	setModelMatrix();

	// draw all meshes assigned to this node
	for (unsigned int n = 0; n < nd->mNumMeshes; ++n)
	{
		// bind material uniform
		mShaderProgram.setUniformValue("material", QMatrix4x4(myMeshes[nd->mMeshes[n]].material.material));
		mShaderProgram.setUniformValue("texCount", (GLfloat)myMeshes[nd->mMeshes[n]].material.texCount);
		mShaderProgram.setUniformValue("shininess", myMeshes[nd->mMeshes[n]].material.shininess);
		// bind texture
		glBindTexture(GL_TEXTURE_2D, myMeshes[nd->mMeshes[n]].texIndex);
		// bind VAO
		myMeshes[nd->mMeshes[n]].mVAO->bind();
		// draw
		glDrawElements(GL_TRIANGLES, myMeshes[nd->mMeshes[n]].numFaces * 3, GL_UNSIGNED_INT, 0);
		//myMeshes[nd->mMeshes[n]].mVAO->release();
		glCheckError();
	}

	// draw all children
	for (unsigned int n = 0; n < nd->mNumChildren; ++n)
	{
		recursive_render(sc, nd->mChildren[n]);
	}
	popMatrix();
}

void QtOpenGLScene::renderScene(int rotX, int rotY, int rotZ)
{
	// Use our shader:
	mShaderProgram.bind();
	// Set camera matrix:
	setCamera(camX, camY, camZ, 0, 0, 0);
	// Set the model matrix to the identity Matrix:
	setIdentityMatrix(modelMatrix, 4);
	// Sets the model matrix to a scale matrix so that the model fits in the window:
	scale(assimp_manager->getScaleFactor(), assimp_manager->getScaleFactor(), assimp_manager->getScaleFactor());
	// Rotate the model:
	rotate((float)rotX, 1.0f, 0.0f, 0.0f);
	rotate((float)rotY, 0.0f, 1.0f, 0.0f);
	rotate((float)rotZ, 0.0f, 0.0f, 1.0f);
	// We are only going to use texture unit 0.
	// Unfortunately samplers can't reside in uniform blocks
	// so we have set this uniform separately:
	mShaderProgram.setUniformValue(texUnit, 0);
	recursive_render(assimp_manager->getScene(), assimp_manager->getScene()->mRootNode);
	// FPS computation and display:
	frame++;
}
