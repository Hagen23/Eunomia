#ifndef QTOPENGLSCENE_H
#define QTOPENGLSCENE_H

#include <string>
#include "cMacros.h"
#include "abstractscene.h"

#include <qdatetime.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QObject>
#include <QOpenGLContext>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
    virtual void render(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int fovY);
    virtual void resize(int width, int height);
	
	virtual int load_ANCONEUS();
	virtual int pack_ANCONEUS();

	virtual int load_BRACHIALIS();
	virtual int pack_BRACHIALIS();

	virtual int load_BRACHIORADIALIS();
	virtual int pack_BRACHIORADIALIS();

	virtual int load_PRONATOR_TERES();
	virtual int pack_PRONATOR_TERES();

	virtual int load_BICEPS_BRACHII();
	virtual int pack_BICEPS_BRACHII();

	virtual int load_TRICEPS_BRACHII();
	virtual int pack_TRICEPS_BRACHII();

	virtual int load_OTHER();
	virtual int pack_OTHER();

	virtual int load_BONES();
	virtual int pack_BONES();

	virtual void toggle_ANCONEUS(bool v);
	virtual void toggle_BRACHIALIS(bool v);
	virtual void toggle_BRACHIORADIALIS(bool v);
	virtual void toggle_PRONATOR_TERES(bool v);
	virtual void toggle_BICEPS_BRACHII(bool v);
	virtual void toggle_TRICEPS_BRACHII(bool v);
	virtual void toggle_OTHER(bool v);
	virtual void toggle_BONES(bool v);
	virtual void toggle_GRID(bool v);
	virtual void toggle_AXES(bool v);

	virtual void toggle_ANCONEUS_wireframe(bool v);
	virtual void toggle_BRACHIALIS_wireframe(bool v);
	virtual void toggle_BRACHIORADIALIS_wireframe(bool v);
	virtual void toggle_PRONATOR_TERES_wireframe(bool v);
	virtual void toggle_BICEPS_BRACHII_wireframe(bool v);
	virtual void toggle_TRICEPS_BRACHII_wireframe(bool v);
	virtual void toggle_OTHER_wireframe(bool v);
	virtual void toggle_BONES_wireframe(bool v);

	virtual void set_ANCONEUS_opacity(float o);
	virtual void set_BRACHIALIS_opacity(float o);
	virtual void set_BRACHIORADIALIS_opacity(float o);
	virtual void set_PRONATOR_TERES_opacity(float o);
	virtual void set_BICEPS_BRACHII_opacity(float o);
	virtual void set_TRICEPS_BRACHII_opacity(float o);
	virtual void set_OTHER_opacity(float o);
	virtual void set_BONES_opacity(float o);

	virtual void setXPivot(float v);
	virtual void setYPivot(float v);
	virtual void setZPivot(float v);

private:

	struct MODEL_SECTION
	{
		float					model_opacity;
		bool					model_loaded;
		bool					model_shown;
		bool					model_wireframe;
		std::string				model_path;
		std::string				model_base_path;
		AssimpManager*			model_manager;
		TextureManager*			texture_manager;
		std::vector<AssimpMesh>	model_meshes;
	};

	void prepareShaderProgram();
	void prepareVertexBuffers();
	int genVAOsAndUniformBuffer(const aiScene* sc, vector<AssimpMesh>&	meshes, TextureManager* tm);
	void set_float4(float f[4], float a, float b, float c, float d);
	void color4_to_float4(const aiColor4D* c, float f[4]);
	void recursive_render(const aiScene* sc, const aiNode* nd, vector<AssimpMesh>&	meshes, float opacity);
	void pushMatrix();
	void popMatrix();
	void setRotationMatrix(glm::mat4& mat, float angle, float x, float y, float z);
	void setModelMatrix();
	void translate(float x, float y, float z);
	void rotate(float angle, float x, float y, float z);
	void scale(float x, float y, float z);
	void buildProjectionMatrix(float fov, float ratio, float nearp, float farp);
	void setCamera(float posX, float posY, float posZ, float lookAtX, float lookAtY, float lookAtZ);
	void renderScene(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int fovY);

    QOpenGLShaderProgram mShaderProgram;

	vector<MODEL_SECTION>		arm_parts;
	MODEL_SECTION				axes;
	MODEL_SECTION				grid;

	glm::mat4					modelMatrix;
	glm::mat4					viewMatrix;
	glm::mat4					projMatrix;

	// For push and pop matrix
	vector<glm::mat4>			matrixStack;

	// Camera Position
	float camX;
	float camY;
	float camZ;
	float pivotX;
	float pivotY;
	float pivotZ;
	float ratio;
	float fovY;
	float lastFovY;
	float nearP;
	float farP;

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
