#include "QtOpenGLScene.h"
#include "glassert.h"

QtOpenGLScene::QtOpenGLScene() : mShaderProgram()
{
	camX = 0.0f;
	camY = 0.0f;
	camZ = 4.0f;
	pivotX = 0.0f;
	pivotY = 0.0f;
	pivotZ = 0.0f;
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
	axes.model_loaded		= false;
	axes.model_wireframe	= false;
	axes.model_opacity		= 1.0f;
	axes.model_path			= string("assets/arm/axes.obj");
	axes.model_base_path	= StringUtils::getBasePath(axes.model_path);
	axes.texture_manager	= new TextureManager(axes.model_base_path);	
	axes.model_manager		= new AssimpManager(axes.model_path);
	if (!axes.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!axes.texture_manager->LoadGLTextures(axes.model_manager->getScene()))
	{
		return 4;
	}

	grid.model_loaded		= false;
	grid.model_wireframe	= false;
	grid.model_opacity		= 1.0f;
	grid.model_path			= string("assets/arm/grid.obj");
	grid.model_base_path	= StringUtils::getBasePath(grid.model_path);
	grid.texture_manager	= new TextureManager(grid.model_base_path);
	grid.model_manager		= new AssimpManager(grid.model_path);
	if (!grid.model_manager->Import3DFromFile())
	{
		return 3;
	}
	if (!grid.texture_manager->LoadGLTextures(grid.model_manager->getScene()))
	{
		return 4;
	}

	prepareShaderProgram();
	mShaderProgram.bind();
	if (int result = genVAOsAndUniformBuffer(axes.model_manager->getScene(), axes.model_meshes, axes.texture_manager) != 0)
	{
		return result;
	}
	axes.model_loaded		= true;
	axes.model_shown		= true;

	if (int result = genVAOsAndUniformBuffer(grid.model_manager->getScene(), grid.model_meshes, grid.texture_manager) != 0)
	{
		return result;
	}
	grid.model_loaded = true;
	grid.model_shown = true;

	texUnit = mShaderProgram.uniformLocation("texUnit");
	return 0;
}

int QtOpenGLScene::load_ANCONEUS()
{
	MODEL_SECTION anconeus;
	anconeus.model_loaded		= false;
	anconeus.model_wireframe	= false;
	anconeus.model_opacity		= 1.0f;
	anconeus.model_path			= string("assets/arm/anconeus.obj");
	anconeus.model_base_path	= StringUtils::getBasePath(anconeus.model_path);
	anconeus.texture_manager	= new TextureManager(anconeus.model_base_path);
	anconeus.model_manager		= new AssimpManager(anconeus.model_path);
	if (!anconeus.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_ANCONEUS] = anconeus;
	return 0;
}

int QtOpenGLScene::pack_ANCONEUS()
{
	if (!arm_parts[AP_ANCONEUS].texture_manager->LoadGLTextures(arm_parts[AP_ANCONEUS].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_ANCONEUS].model_manager->getScene(), arm_parts[AP_ANCONEUS].model_meshes, arm_parts[AP_ANCONEUS].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_ANCONEUS].model_loaded = true;
	arm_parts[AP_ANCONEUS].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_BRACHIALIS()
{
	MODEL_SECTION brachialis;
	brachialis.model_loaded		= false;
	brachialis.model_wireframe	= false;
	brachialis.model_opacity	= 1.0f;
	brachialis.model_path		= string("assets/arm/brachialis.obj");
	brachialis.model_base_path	= StringUtils::getBasePath(brachialis.model_path);
	brachialis.texture_manager	= new TextureManager(brachialis.model_base_path);
	brachialis.model_manager	= new AssimpManager(brachialis.model_path);
	if (!brachialis.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_BRACHIALIS] = brachialis;
	return 0;
}

int QtOpenGLScene::pack_BRACHIALIS()
{
	if (!arm_parts[AP_BRACHIALIS].texture_manager->LoadGLTextures(arm_parts[AP_BRACHIALIS].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_BRACHIALIS].model_manager->getScene(), arm_parts[AP_BRACHIALIS].model_meshes, arm_parts[AP_BRACHIALIS].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_BRACHIALIS].model_loaded = true;
	arm_parts[AP_BRACHIALIS].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_BRACHIORADIALIS()
{
	MODEL_SECTION brachioradialis;
	brachioradialis.model_loaded	= false;
	brachioradialis.model_wireframe = false;
	brachioradialis.model_opacity	= 1.0f;
	brachioradialis.model_path		= string("assets/arm/brachioradialis.obj");
	brachioradialis.model_base_path	= StringUtils::getBasePath(brachioradialis.model_path);
	brachioradialis.texture_manager	= new TextureManager(brachioradialis.model_base_path);
	brachioradialis.model_manager	= new AssimpManager(brachioradialis.model_path);
	if (!brachioradialis.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_BRACHIORADIALIS] = brachioradialis;
	return 0;
}

int QtOpenGLScene::pack_BRACHIORADIALIS()
{
	if (!arm_parts[AP_BRACHIORADIALIS].texture_manager->LoadGLTextures(arm_parts[AP_BRACHIORADIALIS].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_BRACHIORADIALIS].model_manager->getScene(), arm_parts[AP_BRACHIORADIALIS].model_meshes, arm_parts[AP_BRACHIORADIALIS].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_BRACHIORADIALIS].model_loaded = true;
	arm_parts[AP_BRACHIORADIALIS].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_PRONATOR_TERES()
{
	MODEL_SECTION pronator_teres;
	pronator_teres.model_loaded		= false;
	pronator_teres.model_wireframe	= false;
	pronator_teres.model_opacity	= 1.0f;
	pronator_teres.model_path		= string("assets/arm/pronator_teres.obj");
	pronator_teres.model_base_path	= StringUtils::getBasePath(pronator_teres.model_path);
	pronator_teres.texture_manager	= new TextureManager(pronator_teres.model_base_path);
	pronator_teres.model_manager	= new AssimpManager(pronator_teres.model_path);
	if (!pronator_teres.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_PRONATOR_TERES] = pronator_teres;
	return 0;
}

int QtOpenGLScene::pack_PRONATOR_TERES()
{
	if (!arm_parts[AP_PRONATOR_TERES].texture_manager->LoadGLTextures(arm_parts[AP_PRONATOR_TERES].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_PRONATOR_TERES].model_manager->getScene(), arm_parts[AP_PRONATOR_TERES].model_meshes, arm_parts[AP_PRONATOR_TERES].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_PRONATOR_TERES].model_loaded = true;
	arm_parts[AP_PRONATOR_TERES].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_BICEPS_BRACHII()
{
	MODEL_SECTION biceps_brachii;
	biceps_brachii.model_loaded		= false;
	biceps_brachii.model_wireframe	= false;
	biceps_brachii.model_opacity	= 1.0f;
	biceps_brachii.model_path		= string("assets/arm/biceps_brachii.obj");
	biceps_brachii.model_base_path	= StringUtils::getBasePath(biceps_brachii.model_path);
	biceps_brachii.texture_manager	= new TextureManager(biceps_brachii.model_base_path);
	biceps_brachii.model_manager	= new AssimpManager(biceps_brachii.model_path);
	if (!biceps_brachii.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_BICEPS_BRACHII] = biceps_brachii;
	return 0;
}

int QtOpenGLScene::pack_BICEPS_BRACHII()
{
	if (!arm_parts[AP_BICEPS_BRACHII].texture_manager->LoadGLTextures(arm_parts[AP_BICEPS_BRACHII].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_BICEPS_BRACHII].model_manager->getScene(), arm_parts[AP_BICEPS_BRACHII].model_meshes, arm_parts[AP_BICEPS_BRACHII].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_BICEPS_BRACHII].model_loaded = true;
	arm_parts[AP_BICEPS_BRACHII].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_TRICEPS_BRACHII()
{
	MODEL_SECTION triceps_brachii;
	triceps_brachii.model_loaded	= false;
	triceps_brachii.model_wireframe = false;
	triceps_brachii.model_opacity	= 1.0f;
	triceps_brachii.model_path		= string("assets/arm/triceps_brachii.obj");
	triceps_brachii.model_base_path = StringUtils::getBasePath(triceps_brachii.model_path);
	triceps_brachii.texture_manager = new TextureManager(triceps_brachii.model_base_path);
	triceps_brachii.model_manager	= new AssimpManager(triceps_brachii.model_path);
	if (!triceps_brachii.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_TRICEPS_BRACHII] = triceps_brachii;
	return 0;
}

int QtOpenGLScene::pack_TRICEPS_BRACHII()
{
	if (!arm_parts[AP_TRICEPS_BRACHII].texture_manager->LoadGLTextures(arm_parts[AP_TRICEPS_BRACHII].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_TRICEPS_BRACHII].model_manager->getScene(), arm_parts[AP_TRICEPS_BRACHII].model_meshes, arm_parts[AP_TRICEPS_BRACHII].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_TRICEPS_BRACHII].model_loaded = true;
	arm_parts[AP_TRICEPS_BRACHII].model_shown = true;
	return 0;
}

int QtOpenGLScene::load_OTHER()
{
	MODEL_SECTION other;
	other.model_loaded		= false;
	other.model_wireframe	= false;
	other.model_opacity		= 1.0f;
	other.model_path		= string("assets/arm/other_muscles.obj");
	other.model_base_path	= StringUtils::getBasePath(other.model_path);
	other.texture_manager	= new TextureManager(other.model_base_path);
	other.model_manager		= new AssimpManager(other.model_path);
	if (!other.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_OTHER] = other;
	return 0;
}

int QtOpenGLScene::pack_OTHER()
{
	if (!arm_parts[AP_OTHER].texture_manager->LoadGLTextures(arm_parts[AP_OTHER].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_OTHER].model_manager->getScene(), arm_parts[AP_OTHER].model_meshes, arm_parts[AP_OTHER].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_OTHER].model_loaded = true;
	arm_parts[AP_OTHER].model_shown = true;

	return 0;
}

int QtOpenGLScene::load_BONES()
{
	MODEL_SECTION bones;
	bones.model_loaded = false;
	bones.model_wireframe = false;
	bones.model_opacity = 1.0f;
	bones.model_path = string("assets/arm/bones.obj");
	bones.model_base_path = StringUtils::getBasePath(bones.model_path);
	bones.texture_manager = new TextureManager(bones.model_base_path);
	bones.model_manager = new AssimpManager(bones.model_path);
	if (!bones.model_manager->Import3DFromFile())
	{
		return 3;
	}
	arm_parts[AP_BONES] = bones;
	return 0;
}

int QtOpenGLScene::pack_BONES()
{
	if (!arm_parts[AP_BONES].texture_manager->LoadGLTextures(arm_parts[AP_BONES].model_manager->getScene()))
	{
		return 4;
	}
	if (int result = genVAOsAndUniformBuffer(arm_parts[AP_BONES].model_manager->getScene(), arm_parts[AP_BONES].model_meshes, arm_parts[AP_BONES].texture_manager) != 0)
	{
		return result;
	}
	arm_parts[AP_BONES].model_loaded = true;
	arm_parts[AP_BONES].model_shown = true;

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

void QtOpenGLScene::toggle_BRACHIORADIALIS(bool v)
{
	arm_parts[AP_BRACHIORADIALIS].model_shown = v;
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

void QtOpenGLScene::toggle_BONES(bool v)
{
	arm_parts[AP_BONES].model_shown = v;
}

void QtOpenGLScene::toggle_GRID(bool v)
{
	grid.model_shown = v;
}

void QtOpenGLScene::toggle_AXES(bool v)
{
	axes.model_shown = v;
}

void QtOpenGLScene::toggle_ANCONEUS_wireframe(bool v)
{
	arm_parts[AP_ANCONEUS].model_wireframe = v;
}

void QtOpenGLScene::toggle_BRACHIALIS_wireframe(bool v)
{
	arm_parts[AP_BRACHIALIS].model_wireframe = v;
}

void QtOpenGLScene::toggle_BRACHIORADIALIS_wireframe(bool v)
{
	arm_parts[AP_BRACHIORADIALIS].model_wireframe = v;
}

void QtOpenGLScene::toggle_PRONATOR_TERES_wireframe(bool v)
{
	arm_parts[AP_PRONATOR_TERES].model_wireframe = v;
}

void QtOpenGLScene::toggle_BICEPS_BRACHII_wireframe(bool v)
{
	arm_parts[AP_BICEPS_BRACHII].model_wireframe = v;
}

void QtOpenGLScene::toggle_TRICEPS_BRACHII_wireframe(bool v)
{
	arm_parts[AP_TRICEPS_BRACHII].model_wireframe = v;
}

void QtOpenGLScene::toggle_OTHER_wireframe(bool v)
{
	arm_parts[AP_OTHER].model_wireframe = v;
}

void QtOpenGLScene::toggle_BONES_wireframe(bool v)
{
	arm_parts[AP_BONES].model_wireframe = v;
}

void QtOpenGLScene::set_ANCONEUS_opacity(float o)
{
	arm_parts[AP_ANCONEUS].model_opacity = o;
}

void QtOpenGLScene::set_BRACHIALIS_opacity(float o)
{
	arm_parts[AP_BRACHIALIS].model_opacity = o;
}

void QtOpenGLScene::set_BRACHIORADIALIS_opacity(float o)
{
	arm_parts[AP_BRACHIORADIALIS].model_opacity = o;
}

void QtOpenGLScene::set_PRONATOR_TERES_opacity(float o)
{
	arm_parts[AP_PRONATOR_TERES].model_opacity = o;
}

void QtOpenGLScene::set_BICEPS_BRACHII_opacity(float o)
{
	arm_parts[AP_BICEPS_BRACHII].model_opacity = o;
}

void QtOpenGLScene::set_TRICEPS_BRACHII_opacity(float o)
{
	arm_parts[AP_TRICEPS_BRACHII].model_opacity = o;
}

void QtOpenGLScene::set_OTHER_opacity(float o)
{
	arm_parts[AP_OTHER].model_opacity = o;
}

void QtOpenGLScene::set_BONES_opacity(float o)
{
	arm_parts[AP_BONES].model_opacity = o;
}

void QtOpenGLScene::setXPivot(float v)
{
	pivotX = v;
}

void QtOpenGLScene::setYPivot(float v)
{
	pivotY = v;
}

void QtOpenGLScene::setZPivot(float v)
{
	pivotZ = v;
}

void QtOpenGLScene::update(float t)
{
    Q_UNUSED(t);
}

void QtOpenGLScene::render(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int fovY)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(30.0f / 255.0f, 30.0f / 255.0f, 30.0f / 255.0f, 1.0f);
	renderScene(rotModelX, rotModelY, rotModelZ, posCamX, posCamY, posCamZ, fovY);
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

int QtOpenGLScene::genVAOsAndUniformBuffer(const aiScene* sc, vector<AssimpMesh>& meshes, TextureManager* tm)
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
		if (!aMesh.mVAO->create())
		{
			return 1;
		}
		aMesh.mVAO->bind();
		// buffer for faces
		if (!aMesh.mFaceIndicesBuffer.create())
		{
			return 2;
		}
		aMesh.mFaceIndicesBuffer.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
		aMesh.mFaceIndicesBuffer.bind();
		aMesh.mFaceIndicesBuffer.allocate(faceArray, aMesh.numFaces * 3 * sizeof(unsigned int));
		// buffer for vertex positions
		if (mesh->HasPositions())
		{
			if (!aMesh.mVertexPositionBuffer.create())
			{
				return 3;
			}
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
			if (!aMesh.mVertexNormalBuffer.create())
			{
				return 4;
			}
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

			if (!aMesh.mVertexTextureBuffer.create())
			{
				return 5;
			}
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
			aMesh.texIndex = tm->textureIdMap[texPath.data];
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
	return 0;
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
	//mShaderProgram.bind();
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
	//mShaderProgram.bind();
	mShaderProgram.setUniformValue("viewMatrix", vm.transposed());
}

void QtOpenGLScene::recursive_render(const aiScene *sc, const aiNode* nd, vector<AssimpMesh>& meshes, float opacity)
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
		mShaderProgram.setUniformValue("opacity", opacity);
		// bind texture
		glBindTexture(GL_TEXTURE_2D, meshes[nd->mMeshes[n]].texIndex);
		// bind VAO
		meshes[nd->mMeshes[n]].mVAO->bind();
		// draw
		glDrawElements(GL_TRIANGLES, meshes[nd->mMeshes[n]].numFaces * 3, GL_UNSIGNED_INT, 0);
		meshes[nd->mMeshes[n]].mVAO->release();
		glCheckError();
	}

	// draw all children
	for (unsigned int n = 0; n < nd->mNumChildren; ++n)
	{
		recursive_render(sc, nd->mChildren[n], meshes, opacity);
	}
	popMatrix();
}

void QtOpenGLScene::renderScene(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int _fovY)
{
	// Use our shader:
	mShaderProgram.bind();
	// Set camera matrix:
	camX = (float)posCamX;
	camY = (float)posCamY;
	camZ = (float)posCamZ;
	fovY = 44.0f - ((float)_fovY/1000.0f);
	buildProjectionMatrix(fovY, ratio, nearP, farP);
	setCamera(camX, camY, camZ, pivotX, pivotY, pivotZ);
	// Set the model matrix to the identity Matrix:
	modelMatrix = glm::mat4(1.0);

	// Sets the model matrix to a scale matrix so that the model fits in the window:
	float sf = grid.model_manager->getScaleFactor();
	scale(sf, sf, sf);
	// Unfortunately samplers can't reside in uniform blocks
	// so we have set this uniform separately:
	mShaderProgram.setUniformValue(texUnit, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (grid.model_shown)
		recursive_render(grid.model_manager->getScene(), grid.model_manager->getScene()->mRootNode, grid.model_meshes, grid.model_opacity);
	if (axes.model_shown)
		recursive_render(axes.model_manager->getScene(), axes.model_manager->getScene()->mRootNode, axes.model_meshes, axes.model_opacity);

	// Rotate the model:
	rotate((float)rotModelX, 1.0f, 0.0f, 0.0f);
	rotate((float)rotModelY, 0.0f, 1.0f, 0.0f);
	rotate((float)rotModelZ, 0.0f, 0.0f, 1.0f);
	for (unsigned int i = 0; i < NUM_ARM_PARTS; i++)
	{
		if (arm_parts[i].model_loaded && arm_parts[i].model_shown)
		{
			if (arm_parts[i].model_wireframe)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				glLineWidth(0.5f);
			}
			else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			recursive_render(arm_parts[i].model_manager->getScene(), arm_parts[i].model_manager->getScene()->mRootNode, arm_parts[i].model_meshes, arm_parts[i].model_opacity);
			glLineWidth(1.0f);
		}
	}
	mShaderProgram.release();
	frame++;
}
