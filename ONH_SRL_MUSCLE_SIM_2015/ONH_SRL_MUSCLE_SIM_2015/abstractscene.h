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
	virtual void render(int rotModelX, int rotModelY, int rotModelZ, int posCamX, int posCamY, int posCamZ, int fovY, int opacity) = 0;
	virtual void resize(int width, int height) = 0;
	virtual int load_ANCONEUS() = 0;
	virtual int load_BRACHIALIS() = 0;
	virtual int load_BRACHIORDIALIS() = 0;
	virtual int load_PRONATOR_TERES() = 0;
	virtual int load_BICEPS_BRACHII() = 0;
	virtual int load_TRICEPS_BRACHII() = 0;
	virtual int load_OTHER() = 0;
	virtual void toggle_ANCONEUS(bool v) = 0;
	virtual void toggle_BRACHIALIS(bool v) = 0;
	virtual void toggle_BRACHIORDIALIS(bool v) = 0;
	virtual void toggle_PRONATOR_TERES(bool v) = 0;
	virtual void toggle_BICEPS_BRACHII(bool v) = 0;
	virtual void toggle_TRICEPS_BRACHII(bool v) = 0;
	virtual void toggle_OTHER(bool v) = 0;

protected:
	QOpenGLContext *mContext;
};

#endif // ABSTRACTSCENE_H
