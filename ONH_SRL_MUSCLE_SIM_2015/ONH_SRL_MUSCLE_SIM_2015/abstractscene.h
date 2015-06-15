#ifndef ABSTRACTSCENE_H
#define ABSTRACTSCENE_H

class QOpenGLContext;

class AbstractScene
{
public:
	AbstractScene() : mContext(0) {}
	virtual ~AbstractScene(){}

	void setContext(QOpenGLContext *context) { mContext = context; }
	QOpenGLContext* context() const { return mContext; }

	virtual int initialize() = 0;
	virtual void update(float t) = 0;
	virtual void render(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int fovY) = 0;
	virtual void resize(int width, int height) = 0;
	
	virtual int load_ANCONEUS() = 0;
	virtual int pack_ANCONEUS() = 0;

	virtual int load_BRACHIALIS() = 0;
	virtual int pack_BRACHIALIS() = 0;

	virtual int load_BRACHIORADIALIS() = 0;
	virtual int pack_BRACHIORADIALIS() = 0;
	
	virtual int load_PRONATOR_TERES() = 0;
	virtual int pack_PRONATOR_TERES() = 0;
	
	virtual int load_BICEPS_BRACHII() = 0;
	virtual int pack_BICEPS_BRACHII() = 0;
	
	virtual int load_TRICEPS_BRACHII() = 0;
	virtual int pack_TRICEPS_BRACHII() = 0;
	
	virtual int load_OTHER() = 0;
	virtual int pack_OTHER() = 0;

	virtual int load_BONES() = 0;
	virtual int pack_BONES() = 0;

	virtual void toggle_ANCONEUS(bool v) = 0;
	virtual void toggle_BRACHIALIS(bool v) = 0;
	virtual void toggle_BRACHIORADIALIS(bool v) = 0;
	virtual void toggle_PRONATOR_TERES(bool v) = 0;
	virtual void toggle_BICEPS_BRACHII(bool v) = 0;
	virtual void toggle_TRICEPS_BRACHII(bool v) = 0;
	virtual void toggle_OTHER(bool v) = 0;
	virtual void toggle_BONES(bool v) = 0;
	virtual void toggle_GRID(bool v) = 0;
	virtual void toggle_AXES(bool v) = 0;

	virtual void toggle_ANCONEUS_wireframe(bool v) = 0;
	virtual void toggle_BRACHIALIS_wireframe(bool v) = 0;
	virtual void toggle_BRACHIORADIALIS_wireframe(bool v) = 0;
	virtual void toggle_PRONATOR_TERES_wireframe(bool v) = 0;
	virtual void toggle_BICEPS_BRACHII_wireframe(bool v) = 0;
	virtual void toggle_TRICEPS_BRACHII_wireframe(bool v) = 0;
	virtual void toggle_OTHER_wireframe(bool v) = 0;
	virtual void toggle_BONES_wireframe(bool v) = 0;

	virtual void set_ANCONEUS_opacity(float o) = 0;
	virtual void set_BRACHIALIS_opacity(float o) = 0;
	virtual void set_BRACHIORADIALIS_opacity(float o) = 0;
	virtual void set_PRONATOR_TERES_opacity(float o) = 0;
	virtual void set_BICEPS_BRACHII_opacity(float o) = 0;
	virtual void set_TRICEPS_BRACHII_opacity(float o) = 0;
	virtual void set_OTHER_opacity(float o) = 0;
	virtual void set_BONES_opacity(float o) = 0;

	virtual void setXPivot(float v) = 0;
	virtual void setYPivot(float v) = 0;
	virtual void setZPivot(float v) = 0;

protected:
	QOpenGLContext *mContext;
};

#endif // ABSTRACTSCENE_H
