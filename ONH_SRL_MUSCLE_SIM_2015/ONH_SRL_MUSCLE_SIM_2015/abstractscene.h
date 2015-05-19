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

protected:
	QOpenGLContext *mContext;
};

#endif // ABSTRACTSCENE_H
