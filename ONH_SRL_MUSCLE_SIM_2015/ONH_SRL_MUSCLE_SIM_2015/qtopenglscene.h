#ifndef QTOPENGLSCENE_H
#define QTOPENGLSCENE_H

#include <string>
#include "abstractscene.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QObject>
#include <QOpenGLContext>

#include "cAssimpMesh.h"
#include "cAssimpManager.h"
#include "cTextureManager.h"

using namespace std;

class QtOpenGLScene : public AbstractScene
{
public:
    QtOpenGLScene();

    virtual int initialize();
    virtual void update(float t);
    virtual void render(int rotX, int rotY, int rotZ);
    virtual void resize(int width, int height);

private:

	void prepareShaderProgram();
	void prepareVertexBuffers();
	void genVAOsAndUniformBuffer(const aiScene* sc);
	void set_float4(float f[4], float a, float b, float c, float d);
	void color4_to_float4(const aiColor4D* c, float f[4]);
	void crossProduct(float* a, float* b, float* res);
	void normalize(float* a);
	void recursive_render(const aiScene* sc, const aiNode* nd);
	void pushMatrix();
	void popMatrix();
	void setIdentityMatrix(float* mat, int size);
	void multMatrix(float* a, float* b);
	void setTranslationMatrix(float* mat, float x, float y, float z);
	void setScaleMatrix(float* mat, float sx, float sy, float sz);
	void setRotationMatrix(float* mat, float angle, float x, float y, float z);
	void setModelMatrix();
	void translate(float x, float y, float z);
	void rotate(float angle, float x, float y, float z);
	void scale(float x, float y, float z);
	void buildProjectionMatrix(float fov, float ratio, float nearp, float farp);
	void setCamera(float posX, float posY, float posZ, float lookAtX, float lookAtY, float lookAtZ);
	void renderScene(int rotX, int rotY, int rotZ);

    QOpenGLShaderProgram mShaderProgram;

	//Custom objects:
	AssimpManager*		assimp_manager;
	TextureManager*		tex_manager;
	vector<AssimpMesh>	myMeshes;

	// Model Matrix (part of the OpenGL Model View Matrix)
	float modelMatrix[16];
	float projMatrix[16];
	float viewMatrix[16];

	// For push and pop matrix
	vector<float*> matrixStack;

	// Replace the model name by your model's filename
	string modelname;

	// Camera Position
	float camX = 0, camY = 0, camZ = 10;

	// Mouse Tracking Variables
	int startX, startY, tracking = 0;

	// Camera Spherical Coordinates
	float alpha = 0.0f, beta = 0.0f;
	float r = 5.0f;

	// Frame counting and FPS computation
	long time, timebase = 0, frame = 0;
	char s[32];

	int texUnit;
};

#endif // QtOpenGLScene_H
