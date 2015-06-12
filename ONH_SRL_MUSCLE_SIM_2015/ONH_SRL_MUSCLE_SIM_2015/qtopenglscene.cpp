#include "QtOpenGLScene.h"
#include "glassert.h"

QtOpenGLScene::QtOpenGLScene() : mShaderProgram()
{
	camX = 0.0f;
	camY = 0.0f;
	camZ = 4.0f;
	fovY = 44.0f;
	lastFovY = fovY;
	nearP = 0.09f;
	farP = 100.0f;
	ratio = 4 / 3.0f;

	arm_parts.resize(NUM_ARM_PARTS);
	for (unsigned int i = 0; i < NUM_ARM_PARTS; i++)
	{
		arm_parts[i].model_loaded = false;
		arm_parts[i].model_shown = true;
	}
}

int QtOpenGLScene::initialize()
{
	//modelname = "assets/bench.obj";
	//modelname = "assets/sphere.obj";
	//modelname = "assets/sphere_notex.obj";
	//modelname = "assets/spider.obj";
	//modelname = "assets/testmixed.obj";
	//modelname = "assets/WusonOBJ.obj";


	axes.model_path = string("assets/arm/axes.obj");
	axes.model_loaded = false;
	axes.model_manager = new AssimpManager(axes.model_path);
	if (!axes.model_manager->Import3DFromFile())
	{
		return 3;
	}
	string basepath = StringUtils::getBasePath(axes.model_path);
	tex_manager = new TextureManager(basepath);
	if (!tex_manager->LoadGLTextures(axes.model_manager->getScene()))
	{
		return 4;
	}
	axes.model_loaded = true;
	axes.model_shown = true;

	prepareShaderProgram();

	genVAOsAndUniformBuffer(axes.model_manager->getScene(), axes.model_meshes);

	mShaderProgram.bind();
	texUnit = mShaderProgram.uniformLocation("texUnit");
	return 0;
}

int QtOpenGLScene::load_ANCONEUS()
{
	//ANCONEUS
	MODEL_SECTION anconeus;
	anconeus.model_loaded = false;
	anconeus.model_path = string("assets/arm/anconeus.obj");
	anconeus.model_manager = new AssimpManager(anconeus.model_path);
	if (!anconeus.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(anconeus.model_manager->getScene()))
	{
		return 4;
	}
	arm_parts[AP_ANCONEUS] = anconeus;
	genVAOsAndUniformBuffer(arm_parts[AP_ANCONEUS].model_manager->getScene(), arm_parts[AP_ANCONEUS].model_meshes);
	arm_parts[AP_ANCONEUS].model_loaded = true;
	arm_parts[AP_ANCONEUS].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_BRACHIALIS()
{
	//BRACHIALIS
	MODEL_SECTION brachialis;
	brachialis.model_loaded = false;
	brachialis.model_path = string("assets/arm/brachialis.obj");
	brachialis.model_manager = new AssimpManager(brachialis.model_path);
	if (!brachialis.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(brachialis.model_manager->getScene()))
	{
		return 4;
	}
	arm_parts[AP_BRACHIALIS] = brachialis;
	genVAOsAndUniformBuffer(arm_parts[AP_BRACHIALIS].model_manager->getScene(), arm_parts[AP_BRACHIALIS].model_meshes);
	arm_parts[AP_BRACHIALIS].model_loaded = true;
	arm_parts[AP_BRACHIALIS].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_BRACHIORDIALIS()
{
	//BRACHIORDIALIS
	MODEL_SECTION brachiordialis;
	brachiordialis.model_loaded = false;
	brachiordialis.model_path = string("assets/arm/brachiordialis.obj");
	brachiordialis.model_manager = new AssimpManager(brachiordialis.model_path);
	if (!brachiordialis.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(brachiordialis.model_manager->getScene()))
	{
		return 4;
	}
	arm_parts[AP_BRACHIORDIALIS] = brachiordialis;
	genVAOsAndUniformBuffer(arm_parts[AP_BRACHIORDIALIS].model_manager->getScene(), arm_parts[AP_BRACHIORDIALIS].model_meshes);
	arm_parts[AP_BRACHIORDIALIS].model_loaded = true;
	arm_parts[AP_BRACHIORDIALIS].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_PRONATOR_TERES()
{
	//PRONATOR_TERES
	MODEL_SECTION pronator_teres;
	pronator_teres.model_loaded = false;
	pronator_teres.model_path = string("assets/arm/pronator_teres.obj");
	pronator_teres.model_manager = new AssimpManager(pronator_teres.model_path);
	if (!pronator_teres.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(pronator_teres.model_manager->getScene()))
	{
		return 4;
	}
	arm_parts[AP_PRONATOR_TERES] = pronator_teres;
	genVAOsAndUniformBuffer(arm_parts[AP_PRONATOR_TERES].model_manager->getScene(), arm_parts[AP_PRONATOR_TERES].model_meshes);
	arm_parts[AP_PRONATOR_TERES].model_loaded = true;
	arm_parts[AP_PRONATOR_TERES].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_BICEPS_BRACHII()
{
	//BICEPS_BRACHII
	MODEL_SECTION biceps_brachii;
	biceps_brachii.model_loaded = false;
	biceps_brachii.model_path = string("assets/arm/biceps_brachii.obj");
	biceps_brachii.model_manager = new AssimpManager(biceps_brachii.model_path);
	if (!biceps_brachii.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(biceps_brachii.model_manager->getScene()))
	{
		return 4;
	}
	arm_parts[AP_BICEPS_BRACHII] = biceps_brachii;
	genVAOsAndUniformBuffer(arm_parts[AP_BICEPS_BRACHII].model_manager->getScene(), arm_parts[AP_BICEPS_BRACHII].model_meshes);
	arm_parts[AP_BICEPS_BRACHII].model_loaded = true;
	arm_parts[AP_BICEPS_BRACHII].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_TRICEPS_BRACHII()
{
	//TRICEPS_BRACHII
	MODEL_SECTION triceps_brachii;
	triceps_brachii.model_loaded = false;
	triceps_brachii.model_path = string("assets/arm/triceps_brachii.obj");
	triceps_brachii.model_manager = new AssimpManager(triceps_brachii.model_path);
	if (!triceps_brachii.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(triceps_brachii.model_manager->getScene()))
	{
		return 4;
	}
	arm_parts[AP_TRICEPS_BRACHII] = triceps_brachii;
	genVAOsAndUniformBuffer(arm_parts[AP_TRICEPS_BRACHII].model_manager->getScene(), arm_parts[AP_TRICEPS_BRACHII].model_meshes);
	arm_parts[AP_TRICEPS_BRACHII].model_loaded = true;
	arm_parts[AP_TRICEPS_BRACHII].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_OTHER()
{
	//OTHER
	MODEL_SECTION other;
	other.model_loaded = false;
	other.model_path = string("assets/arm/other.obj");
	other.model_manager = new AssimpManager(other.model_path);
	if (!other.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!tex_manager->LoadGLTextures(other.model_manager->getScene()))
	{
		return 4;
	}
	arm_parts[AP_OTHER] = other;
	genVAOsAndUniformBuffer(arm_parts[AP_OTHER].model_manager->getScene(), arm_parts[AP_OTHER].model_meshes);
	arm_parts[AP_OTHER].model_loaded = true;
	arm_parts[AP_OTHER].model_shown = true;
	return 0;
}

void QtOpenGLScene::toggle_ANCONEUS(bool v)
{
	arm_parts[AP_ANCONEUS].model_shown = v;
}

void QtOpenGLScene::toggle_BRACHIALIS(bool v)
{
	arm_parts[AP_BRACHIALIS].model_shown = v;
}

void QtOpenGLScene::toggle_BRACHIORDIALIS(bool v)
{
	arm_parts[AP_BRACHIORDIALIS].model_shown = v;
}

void QtOpenGLScene::toggle_PRONATOR_TERES(bool v)
{
	arm_parts[AP_PRONATOR_TERES].model_shown = v;
}

void QtOpenGLScene::toggle_BICEPS_BRACHII(bool v)
{
	arm_parts[AP_BICEPS_BRACHII].model_shown = v;
}

void QtOpenGLScene::toggle_TRICEPS_BRACHII(bool v)
{
	arm_parts[AP_TRICEPS_BRACHII].model_shown = v;
}

void QtOpenGLScene::toggle_OTHER(bool v)
{
	arm_parts[AP_OTHER].model_shown = v;
}


void QtOpenGLScene::update(float t)
{
    Q_UNUSED(t);
}

void QtOpenGLScene::render(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int fovY, int opacity)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(30.0f / 255.0f, 30.0f / 255.0f, 30.0f / 255.0f, 1.0f);
	renderScene(rotModelX, rotModelY, rotModelZ, posCamX, posCamY, posCamZ, fovY, opacity);
}

void QtOpenGLScene::resize(int width, int height)
{
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (height == 0) height = 1;
	// Set the viewport to be the entire window
	glAssert(glViewport(0, 0, width, height));
	ratio = (1.0f * width) / height;
	buildProjectionMatrix(fovY, ratio, nearP, farP);
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

void QtOpenGLScene::genVAOsAndUniformBuffer(const aiScene* sc, vector<AssimpMesh>&	meshes)
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

		qDebug() << "LOADING MESH: " << QString(mesh->mName.C_Str());

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

		aiString name;
		mtl->Get(AI_MATKEY_NAME, name);
		qDebug() << "MATERIAL: " << name.C_Str();

		float c[4];
		set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
		aiColor4D diffuse;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
			color4_to_float4(&diffuse, c);
		memcpy(aMesh.material.diffuse, c, sizeof(c));

		float opacity;
		if (AI_SUCCESS == aiGetMaterialFloat(mtl, AI_MATKEY_OPACITY, &opacity))
		{
			qDebug() << "OPACITY =" << opacity;
			aMesh.material.opacity = opacity;
		}
		
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
		meshes.push_back(aMesh);
	}
	glCheckError();
}

void QtOpenGLScene::pushMatrix()
{
	glm::mat4 aux = modelMatrix;
	matrixStack.push_back(aux);
}

void QtOpenGLScene::popMatrix()
{
	glm::mat4 m = matrixStack[matrixStack.size() - 1];
	modelMatrix = m;
	matrixStack.pop_back();
}

void QtOpenGLScene::setModelMatrix()
{
	QMatrix4x4 mm = QMatrix4x4(&modelMatrix[0][0]);
	//qDebug() << "NEW MODEL MATRIX:";
	//qDebug() << mm;
	mShaderProgram.bind();
	mShaderProgram.setUniformValue("modelMatrix", mm.transposed());
}

void QtOpenGLScene::translate(float x, float y, float z)
{
	modelMatrix = glm::translate(modelMatrix, glm::vec3(x, y, z));
	setModelMatrix();
}

void QtOpenGLScene::rotate(float angle, float x, float y, float z)
{
	modelMatrix = glm::rotate(modelMatrix, (float)DEG2RAD*angle, glm::vec3(x, y, z));
	setModelMatrix();
}

void QtOpenGLScene::scale(float x, float y, float z)
{
	modelMatrix = glm::scale(modelMatrix, glm::vec3(x,y,z));
	setModelMatrix();
}

void QtOpenGLScene::buildProjectionMatrix(float fov, float ratio, float nearp, float farp)
{
	projMatrix = glm::perspective(fov, ratio, nearp, farp);

	QMatrix4x4 pm = QMatrix4x4(&projMatrix[0][0]);
	//pm = pm.transposed();
	if (lastFovY != fovY)
	{
		qDebug() << "NEW PROJECTION MATRIX. FOV=" << fov << " RATIO=" << ratio << " NEAR=" << nearp << " FAR=" << farp;
		lastFovY = fovY;
	}
	
	//qDebug() << pm;
	mShaderProgram.bind();
	mShaderProgram.setUniformValue("projMatrix", pm.transposed());
	glCheckError();
}

void QtOpenGLScene::setCamera(float posX, float posY, float posZ, float lookAtX, float lookAtY, float lookAtZ)
{
	viewMatrix = glm::lookAt(glm::vec3(posX, posY, posZ), glm::vec3(lookAtX, lookAtY, lookAtZ), glm::vec3(0.0f, 1.0f, 0.0f));
	QMatrix4x4 vm = QMatrix4x4(&viewMatrix[0][0]);
	mShaderProgram.bind();
	mShaderProgram.setUniformValue("viewMatrix", vm.transposed());
}

void QtOpenGLScene::recursive_render(const aiScene *sc, const aiNode* nd, vector<AssimpMesh>& meshes)
{
	// Get node transformation matrix
	aiMatrix4x4 m = nd->mTransformation;
	// OpenGL matrices are column major
	m.Transpose();
	// save model matrix and apply node transformation
	pushMatrix();
	glm::mat4 aux(m[0][0]);
	modelMatrix *= aux;
	setModelMatrix();

	// draw all meshes assigned to this node
	for (unsigned int n = 0; n < nd->mNumMeshes; ++n)
	{
		// bind material uniform
		mShaderProgram.setUniformValue("material", meshes[nd->mMeshes[n]].material.material);
		mShaderProgram.setUniformValue("texCount", (GLfloat)meshes[nd->mMeshes[n]].material.texCount);
		mShaderProgram.setUniformValue("shininess", meshes[nd->mMeshes[n]].material.shininess);
		float op = meshes[nd->mMeshes[n]].material.opacity;
		if ( op < 1.0f ) mShaderProgram.setUniformValue("opacity", opacity * op);
		else mShaderProgram.setUniformValue("opacity", op);
		// bind texture
		glBindTexture(GL_TEXTURE_2D, meshes[nd->mMeshes[n]].texIndex);
		// bind VAO
		meshes[nd->mMeshes[n]].mVAO->bind();
		// draw
		glDrawElements(GL_TRIANGLES, meshes[nd->mMeshes[n]].numFaces * 3, GL_UNSIGNED_INT, 0);
		//model_meshes[nd->mMeshes[n]].mVAO->release();
		glCheckError();
	}

	// draw all children
	for (unsigned int n = 0; n < nd->mNumChildren; ++n)
	{
		recursive_render(sc, nd->mChildren[n], meshes);
	}
	popMatrix();
}

void QtOpenGLScene::renderScene(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int _fovY, int _opacity)
{
	// Use our shader:
	mShaderProgram.bind();
	// Set camera matrix:
	camX = (float)posCamX;
	camY = (float)posCamY;
	camZ = (float)posCamZ;
	fovY = 44.0f - ((float)_fovY/1000.0f);
	buildProjectionMatrix(fovY, ratio, nearP, farP);
	setCamera(camX, camY, camZ, 0, 0, 0);
	// Set the model matrix to the identity Matrix:
	modelMatrix = glm::mat4(1.0);
	opacity = ((float)_opacity / 100.0f);
	

	// Sets the model matrix to a scale matrix so that the model fits in the window:
	float sf = axes.model_manager->getScaleFactor();
	scale(sf, sf, sf);
	// Rotate the model:
	//rotate((float)rotModelX, 1.0f, 0.0f, 0.0f);
	//rotate((float)rotModelY, 0.0f, 1.0f, 0.0f);
	//rotate((float)rotModelZ, 0.0f, 0.0f, 1.0f);
	// We are only going to use texture unit 0.
	// Unfortunately samplers can't reside in uniform blocks
	// so we have set this uniform separately:
	mShaderProgram.setUniformValue(texUnit, 0);
	recursive_render(axes.model_manager->getScene(), axes.model_manager->getScene()->mRootNode, axes.model_meshes);


	// Sets the model matrix to a scale matrix so that the model fits in the window:
	//scale(model_manager->getScaleFactor(), model_manager->getScaleFactor(), model_manager->getScaleFactor());
	// Rotate the model:
	rotate((float)rotModelX, 1.0f, 0.0f, 0.0f);
	rotate((float)rotModelY, 0.0f, 1.0f, 0.0f);
	rotate((float)rotModelZ, 0.0f, 0.0f, 1.0f);
	// We are only going to use texture unit 0.
	// Unfortunately samplers can't reside in uniform blocks
	// so we have set this uniform separately:
	//mShaderProgram.setUniformValue(texUnit, 0);
	for (unsigned int i = 0; i < NUM_ARM_PARTS; i++)
	{
		if (arm_parts[i].model_loaded && arm_parts[i].model_shown)
			recursive_render(arm_parts[i].model_manager->getScene(), arm_parts[i].model_manager->getScene()->mRootNode, arm_parts[i].model_meshes);
	}
	// FPS computation and display:
	frame++;
}
