//OPENGL:
#include "glew.h"
#include "freeglut.h"
//DevIL:
#include "il.h"
//Custom:
#include "cStringUtils.h"
#include "cShader.h"
#include "cTextureManager.h"
#include "cAssimpManager.h"
//GLM:
#include "glm/glm.hpp"
//Standard:
#include <math.h>
#include <map>
#include <string>
#include <vector>
//Custom objects:
AssimpManager* assimp_manager;
TextureManager* tex_manager;
Shader* shader;

// Information to render each assimp node
struct MyMesh
{
	GLuint vao;
	GLuint texIndex;
	GLuint uniformBlockIndex;
	int numFaces;
};

vector<struct MyMesh> myMeshes;

// This is for a shader uniform block
struct MyMaterial
{
	float diffuse[4];
	float ambient[4];
	float specular[4];
	float emissive[4];
	float shininess;
	int texCount;
};

// Model Matrix (part of the OpenGL Model View Matrix)
float modelMatrix[16];

// For push and pop matrix
vector<float *> matrixStack;

// Uniform Buffer for Matrices
// this buffer will contain 3 matrices: projection, view and model
// each matrix is a float array with 16 components
GLuint matricesUniBuffer;
#define MatricesUniBufferSize sizeof(float) * 16 * 3
#define ProjMatrixOffset 0
#define ViewMatrixOffset sizeof(float) * 16
#define ModelMatrixOffset sizeof(float) * 16 * 2
#define MatrixSize sizeof(float) * 16

// Shader Names
char *vertexFileName = "shaders/dirlightdiffambpix.vert";
char *fragmentFileName = "shaders/dirlightdiffambpix.frag";

// Replace the model name by your model's filename
//static const string modelname = "assets/bench.obj";
//static const string modelname = "assets/sphere.obj";
//static const string modelname = "assets/sphere_notex.obj";
//static const string modelname = "assets/spider.obj";
//static const string modelname = "assets/testmixed.obj";
//static const string modelname = "assets/WusonOBJ.obj";
static const string modelname = "assets/arm.obj";

// Camera Position
float camX = 0, camY = 0, camZ = 5;

// Mouse Tracking Variables
int startX, startY, tracking = 0;

// Camera Spherical Coordinates
float alpha = 0.0f, beta = 0.0f;
float r = 5.0f;

#define M_PI       3.14159265358979323846f

static inline float
DegToRad(float degrees)
{
	return (float)(degrees * (M_PI / 180.0f));
};

// Frame counting and FPS computation
long time, timebase = 0, frame = 0;
char s[32];

//-----------------------------------------------------------------
// Print for OpenGL errors
//
// Returns 1 if an OpenGL error occurred, 0 otherwise.
//
#define printOpenGLError() printOglError(__FILE__, __LINE__)

int printOglError(char *file, int line)
{
	GLenum glErr;
	int    retCode = 0;

	glErr = glGetError();
	if (glErr != GL_NO_ERROR)
	{
		cout << "ERROR: GLERROR IN FILE '" << file << "' @ LINE " << line << ". MESSAGE: '" << gluErrorString(glErr) << "'" << endl;
		retCode = 1;
	}
	return retCode;
}

// ----------------------------------------------------
// VECTOR STUFF
//

// res = a cross b;
void crossProduct(float *a, float *b, float *res)
{
	res[0] = a[1] * b[2] - b[1] * a[2];
	res[1] = a[2] * b[0] - b[2] * a[0];
	res[2] = a[0] * b[1] - b[0] * a[1];
}

// Normalize a vec3
void normalize(float *a)
{
	float mag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	a[0] /= mag;
	a[1] /= mag;
	a[2] /= mag;
}

// ----------------------------------------------------
// MATRIX STUFF
//

// Push and Pop for modelMatrix
void pushMatrix()
{
	float *aux = (float *)malloc(sizeof(float) * 16);
	memcpy(aux, modelMatrix, sizeof(float) * 16);
	matrixStack.push_back(aux);
}

void popMatrix()
{
	float *m = matrixStack[matrixStack.size() - 1];
	memcpy(modelMatrix, m, sizeof(float) * 16);
	matrixStack.pop_back();
	free(m);
}

// sets the square matrix mat to the identity matrix,
// size refers to the number of rows (or columns)
void setIdentityMatrix(float *mat, int size)
{
	// fill matrix with 0s
	for (int i = 0; i < size * size; ++i)
		mat[i] = 0.0f;

	// fill diagonal with 1s
	for (int i = 0; i < size; ++i)
		mat[i + i * size] = 1.0f;
}

//
// a = a * b;
//
void multMatrix(float *a, float *b)
{
	float res[16];

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			res[j * 4 + i] = 0.0f;
			for (int k = 0; k < 4; ++k) {
				res[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
			}
		}
	}
	memcpy(a, res, 16 * sizeof(float));
}

// Defines a transformation matrix mat with a translation
void setTranslationMatrix(float *mat, float x, float y, float z)
{
	setIdentityMatrix(mat, 4);
	mat[12] = x;
	mat[13] = y;
	mat[14] = z;
}

// Defines a transformation matrix mat with a scale
void setScaleMatrix(float *mat, float sx, float sy, float sz)
{
	setIdentityMatrix(mat, 4);
	mat[0] = sx;
	mat[5] = sy;
	mat[10] = sz;
}

// Defines a transformation matrix mat with a rotation 
// angle alpha and a rotation axis (x,y,z)
void setRotationMatrix(float *mat, float angle, float x, float y, float z)
{
	float radAngle = DegToRad(angle);
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

// ----------------------------------------------------
// Model Matrix 
//
// Copies the modelMatrix to the uniform buffer

void setModelMatrix()
{
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER,
		ModelMatrixOffset, MatrixSize, modelMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

// The equivalent to glTranslate applied to the model matrix
void translate(float x, float y, float z)
{
	float aux[16];
	setTranslationMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glRotate applied to the model matrix
void rotate(float angle, float x, float y, float z)
{
	float aux[16];
	setRotationMatrix(aux, angle, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glScale applied to the model matrix
void scale(float x, float y, float z)
{
	float aux[16];
	setScaleMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// ----------------------------------------------------
// Projection Matrix 
//
// Computes the projection Matrix and stores it in the uniform buffer

void buildProjectionMatrix(float fov, float ratio, float nearp, float farp)
{
	float projMatrix[16];
	float f = 1.0f / tan(fov * (M_PI / 360.0f));
	setIdentityMatrix(projMatrix, 4);

	projMatrix[0] = f / ratio;
	projMatrix[1 * 4 + 1] = f;
	projMatrix[2 * 4 + 2] = (farp + nearp) / (nearp - farp);
	projMatrix[3 * 4 + 2] = (2.0f * farp * nearp) / (nearp - farp);
	projMatrix[2 * 4 + 3] = -1.0f;
	projMatrix[3 * 4 + 3] = 0.0f;

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ProjMatrixOffset, MatrixSize, projMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}


// ----------------------------------------------------
// View Matrix
//
// Computes the viewMatrix and stores it in the uniform buffer
//
// note: it assumes the camera is not tilted, 
// i.e. a vertical up vector along the Y axis (remember gluLookAt?)
//
void setCamera(float posX, float posY, float posZ, float lookAtX, float lookAtY, float lookAtZ)
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

	float viewMatrix[16], aux[16];

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

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ViewMatrixOffset, MatrixSize, viewMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

// ----------------------------------------------------------------------------

void set_float4(float f[4], float a, float b, float c, float d)
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

void color4_to_float4(const aiColor4D *c, float f[4])
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}

void genVAOsAndUniformBuffer(const aiScene* sc)
{
	struct MyMesh aMesh;
	struct MyMaterial aMat;
	GLuint buffer;

	// For each mesh
	for (unsigned int n = 0; n < sc->mNumMeshes; ++n)
	{
		const aiMesh* mesh = sc->mMeshes[n];

		// create array with faces
		// have to convert from Assimp format to array
		unsigned int *faceArray;
		faceArray = (unsigned int *)malloc(sizeof(unsigned int) * mesh->mNumFaces * 3);
		unsigned int faceIndex = 0;

		for (unsigned int t = 0; t < mesh->mNumFaces; ++t) {
			const aiFace* face = &mesh->mFaces[t];

			memcpy(&faceArray[faceIndex], face->mIndices, 3 * sizeof(unsigned int));
			faceIndex += 3;
		}
		aMesh.numFaces = sc->mMeshes[n]->mNumFaces;

		// generate Vertex Array for mesh
		glGenVertexArrays(1, &(aMesh.vao));
		glBindVertexArray(aMesh.vao);

		// buffer for faces
		glGenBuffers(1, &buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mesh->mNumFaces * 3, faceArray, GL_STATIC_DRAW);

		// buffer for vertex positions
		if (mesh->HasPositions()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mVertices, GL_STATIC_DRAW);
			glEnableVertexAttribArray(shader->getVertexLoc());
			glVertexAttribPointer(shader->getVertexLoc(), 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex normals
		if (mesh->HasNormals()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mNormals, GL_STATIC_DRAW);
			glEnableVertexAttribArray(shader->getNormalLoc());
			glVertexAttribPointer(shader->getNormalLoc(), 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex texture coordinates
		if (mesh->HasTextureCoords(0)) {
			float *texCoords = (float *)malloc(sizeof(float) * 2 * mesh->mNumVertices);
			for (unsigned int k = 0; k < mesh->mNumVertices; ++k) {

				texCoords[k * 2] = mesh->mTextureCoords[0][k].x;
				texCoords[k * 2 + 1] = mesh->mTextureCoords[0][k].y;

			}
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * mesh->mNumVertices, texCoords, GL_STATIC_DRAW);
			glEnableVertexAttribArray(shader->getTexCoordLoc());
			glVertexAttribPointer(shader->getTexCoordLoc(), 2, GL_FLOAT, 0, 0, 0);
		}

		// unbind buffers
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// create material uniform buffer
		aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];

		aiString texPath;	//contains filename of texture
		if (AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath)){
			// Bind texture:
			unsigned int texId = tex_manager->textureIdMap[texPath.data];
			aMesh.texIndex = texId;
			aMat.texCount = 1;
		}
		else
			aMat.texCount = 0;

		float c[4];
		set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
		aiColor4D diffuse;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
			color4_to_float4(&diffuse, c);
		memcpy(aMat.diffuse, c, sizeof(c));

		set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
		aiColor4D ambient;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
			color4_to_float4(&ambient, c);
		memcpy(aMat.ambient, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D specular;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
			color4_to_float4(&specular, c);
		memcpy(aMat.specular, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D emission;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
			color4_to_float4(&emission, c);
		memcpy(aMat.emissive, c, sizeof(c));

		float shininess = 0.0;
		unsigned int max;
		aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
		aMat.shininess = shininess;

		glGenBuffers(1, &(aMesh.uniformBlockIndex));
		glBindBuffer(GL_UNIFORM_BUFFER, aMesh.uniformBlockIndex);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(aMat), (void *)(&aMat), GL_STATIC_DRAW);

		myMeshes.push_back(aMesh);
	}
}

// ------------------------------------------------------------
//
// Reshape Callback Function
//
void changeSize(int w, int h)
{
	float ratio;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	ratio = (1.0f * w) / h;
	buildProjectionMatrix(53.13f, ratio, 0.1f, 100.0f);
}

// ------------------------------------------------------------
//
// Render stuff
//

// Render Assimp Model
void recursive_render(const aiScene *sc, const aiNode* nd)
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
		glBindBufferRange(GL_UNIFORM_BUFFER, shader->getMaterialUniLoc(), myMeshes[nd->mMeshes[n]].uniformBlockIndex, 0, sizeof(struct MyMaterial));
		// bind texture
		glBindTexture(GL_TEXTURE_2D, myMeshes[nd->mMeshes[n]].texIndex);
		// bind VAO
		glBindVertexArray(myMeshes[nd->mMeshes[n]].vao);
		// draw
		glDrawElements(GL_TRIANGLES, myMeshes[nd->mMeshes[n]].numFaces * 3, GL_UNSIGNED_INT, 0);
	}

	// draw all children
	for (unsigned int n = 0; n < nd->mNumChildren; ++n)
	{
		recursive_render(sc, nd->mChildren[n]);
	}
	popMatrix();
}

// Rendering Callback Function
void renderScene(void)
{
	static float step = 0.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set camera matrix
	setCamera(camX, camY, camZ, 0, 0, 0);

	// set the model matrix to the identity Matrix
	setIdentityMatrix(modelMatrix, 4);

	// sets the model matrix to a scale matrix so that the model fits in the window
	scale(assimp_manager->getScaleFactor(), assimp_manager->getScaleFactor(), assimp_manager->getScaleFactor());

	// keep rotating the model
	rotate(step, 0.0f, 1.0f, 0.0f);

	// use our shader
	//glUseProgram(program);
	glUseProgram(shader->getProgram());

	// we are only going to use texture unit 0
	// unfortunately samplers can't reside in uniform blocks
	// so we have set this uniform separately
	glUniform1i(shader->getTexUnit(), 0);

	recursive_render(assimp_manager->getScene(), assimp_manager->getScene()->mRootNode);

	// FPS computation and display
	frame++;
	time = glutGet(GLUT_ELAPSED_TIME);
	if (time - timebase > 1000) {
		sprintf_s(s, "FPS:%4.2f",
			frame*1000.0 / (time - timebase));
		timebase = time;
		frame = 0;
		glutSetWindowTitle(s);
	}

	// swap buffers
	glutSwapBuffers();
}

// ------------------------------------------------------------
//
// Events from the Keyboard
//
void processKeys(unsigned char key, int xx, int yy)
{
	switch (key)
	{
	case 27:
		glutLeaveMainLoop();
		break;
	case 'z':
		r -= 0.1f;
		break;
	case 'x':
		r += 0.1f;
		break;
	case 'm':
		glEnable(GL_MULTISAMPLE);
		break;
	case 'n':
		glDisable(GL_MULTISAMPLE);
		break;
	}
	camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r *   						     sin(beta * 3.14f / 180.0f);
}

// ------------------------------------------------------------
//
// Mouse Events
//
void processMouseButtons(int button, int state, int xx, int yy)
{
	// start tracking the mouse
	if (state == GLUT_DOWN)  {
		startX = xx;
		startY = yy;
		if (button == GLUT_LEFT_BUTTON)
			tracking = 1;
		else if (button == GLUT_RIGHT_BUTTON)
			tracking = 2;
	}

	//stop tracking the mouse
	else if (state == GLUT_UP) {
		if (tracking == 1) {
			alpha += (startX - xx);
			beta += (yy - startY);
		}
		else if (tracking == 2) {
			r += (yy - startY) * 0.01f;
		}
		tracking = 0;
	}
}

// Track mouse motion while buttons are pressed
void processMouseMotion(int xx, int yy)
{
	int deltaX, deltaY;
	float alphaAux, betaAux;
	float rAux;

	deltaX = startX - xx;
	deltaY = yy - startY;

	// left mouse button: move camera
	if (tracking == 1) 
	{
		alphaAux = alpha + deltaX;
		betaAux = beta + deltaY;

		if (betaAux > 85.0f)
			betaAux = 85.0f;
		else if (betaAux < -85.0f)
			betaAux = -85.0f;

		rAux = r;

		camX = rAux * cos(betaAux * 3.14f / 180.0f) * sin(alphaAux * 3.14f / 180.0f);
		camZ = rAux * cos(betaAux * 3.14f / 180.0f) * cos(alphaAux * 3.14f / 180.0f);
		camY = rAux * sin(betaAux * 3.14f / 180.0f);
	}
	// right mouse button: zoom
	else if (tracking == 2)
	{
		alphaAux = alpha;
		betaAux = beta;
		rAux = r + (deltaY * 0.01f);

		camX = rAux * cos(betaAux * 3.14f / 180.0f) * sin(alphaAux * 3.14f / 180.0f);
		camZ = rAux * cos(betaAux * 3.14f / 180.0f) * cos(alphaAux * 3.14f / 180.0f);
		camY = rAux * sin(betaAux * 3.14f / 180.0f);
	}
}

void mouseWheel(int wheel, int direction, int x, int y)
{
	r += direction * 0.1f;
	camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r *   						     sin(beta * 3.14f / 180.0f);
}

// ------------------------------------------------------------
//
// Model loading and OpenGL setup
//
int init()
{
	shader					= new Shader(string(vertexFileName), string(fragmentFileName));
	string basepath			= StringUtils::getBasePath(modelname);
	tex_manager				= new TextureManager(basepath);
	assimp_manager			= new AssimpManager(modelname);

	if( !assimp_manager->Import3DFromFile() )
	{
		return 3;
	}
	if( !tex_manager->LoadGLTextures(assimp_manager->getScene()) )
	{
		return 4;
	}

	glGetUniformBlockIndex	= (PFNGLGETUNIFORMBLOCKINDEXPROC)glutGetProcAddress("glGetUniformBlockIndex");
	glUniformBlockBinding	= (PFNGLUNIFORMBLOCKBINDINGPROC)glutGetProcAddress("glUniformBlockBinding");
	glGenVertexArrays		= (PFNGLGENVERTEXARRAYSPROC)glutGetProcAddress("glGenVertexArrays");
	glBindVertexArray		= (PFNGLBINDVERTEXARRAYPROC)glutGetProcAddress("glBindVertexArray");
	glBindBufferRange		= (PFNGLBINDBUFFERRANGEPROC)glutGetProcAddress("glBindBufferRange");
	glDeleteVertexArrays	= (PFNGLDELETEVERTEXARRAYSPROC)glutGetProcAddress("glDeleteVertexArrays");

	shader->compile();
	genVAOsAndUniformBuffer(assimp_manager->getScene());

	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

	//
	// Uniform Block
	//
	glGenBuffers(1, &matricesUniBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferData(GL_UNIFORM_BUFFER, MatricesUniBufferSize, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, shader->getMatricesUniLoc(), matricesUniBuffer, 0, MatricesUniBufferSize);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glEnable(GL_MULTISAMPLE);

	return true;
}

void cleanup(void)
{
	cout << endl;
	cout << "CLEANING UP..." << endl;

	// Delete tex_manager:
	delete tex_manager;
	tex_manager = NULL;

	// Clear myMeshes stuff:
	cout << "CLEARING MESHES..." << endl;
	for (unsigned int i = 0; i < myMeshes.size(); ++i)
	{
		glDeleteVertexArrays(1, &(myMeshes[i].vao));
		glDeleteTextures(1, &(myMeshes[i].texIndex));
		glDeleteBuffers(1, &(myMeshes[i].uniformBlockIndex));
	}
	cout << "DONE." << endl;

	// Delete buffers:
	cout << "DELETING BUFFERS..." << endl;
	glDeleteBuffers(1, &matricesUniBuffer);
	cout << "DONE." << endl;

	// Delete Assimp manager:
	cout << "DELETING ASSIMP MANAGER..." << endl;
	delete assimp_manager;
	assimp_manager = NULL;
	cout << "DONE." << endl;

	// Delete shaders:
	cout << "DELETING SHADERS..." << endl;
	delete shader;
	shader = NULL;
	cout << "DONE." << endl;

	cout << endl;
	cout << "DONE CLEANING UP. BYE." << endl;
	system("pause");
}

// ------------------------------------------------------------
//
// Main function
//
int main(int argc, char* argv[])
{
	// GLUT initialization:
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
	glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_COMPATIBILITY_PROFILE);

	// GLUT window:
	glutInitWindowPosition(100, 10);
	glutInitWindowSize(640, 480);
	glutCreateWindow("OBJ_LOADER");

	// Callback Registration:
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	// Mouse and Keyboard Callbacks:
	glutKeyboardFunc(processKeys);
	glutMouseFunc(processMouseButtons);
	glutMotionFunc(processMouseMotion);
	glutMouseWheelFunc(mouseWheel);

	// Init GLEW:
	//glewExperimental = GL_TRUE;
	glewInit();
	if (glewIsSupported("GL_VERSION_3_3"))
	{
		cout << "READY FOR OPENGL 3.3" << endl;
	}
	else
	{
		cout << "ERROR: OPENGL 3.3 NOT SUPPORTED." << endl;
		cleanup();
		return 1;
	}

	// Init the app (load model and textures) and OpenGL:
	if( !init() )
	{
		cout << "ERROR: COULD NOT LOAD THE MODEL." << endl;
		cleanup();
		return 2;
	}

	cout << "-------------------------------------------------------"	<< endl;
	cout << "CONTEXT:"		<< endl;
	cout << "VENDOR:\t\t"	<< glGetString(GL_VENDOR)					<< endl;
	cout << "RENDERER:\t"	<< glGetString(GL_RENDERER)					<< endl;
	cout << "VERSION:\t"	<< glGetString(GL_VERSION)					<< endl;
	cout << "GLSL:\t\t"		<< glGetString(GL_SHADING_LANGUAGE_VERSION)	<< endl;
	cout << "-------------------------------------------------------"	<< endl;

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoop();

	cleanup();
	return 0;
}
