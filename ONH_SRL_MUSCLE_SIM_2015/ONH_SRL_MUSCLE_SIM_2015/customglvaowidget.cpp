#include "customglvaowidget.h"

static void infoGL()
{
	glCheckError();
	const char *str;
	qDebug() << "\nOpenGL info with GL functions:";
	str = (const char*)glGetString(GL_RENDERER);
	qDebug() << "Renderer : " << QString(str);
	str = (const char*)glGetString(GL_VENDOR);
	qDebug() << "Vendor : " << QString(str);
	str = (const char*)glGetString(GL_VERSION);
	qDebug() << "OpenGL Version : " << QString(str);
	str = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
	qDebug() << "GLSL Version : " << QString(str);
	glCheckError();
}

CustomGLVAOWidget::CustomGLVAOWidget(QScreen *screen) : QWindow(screen), mScene(new QtOpenGLScene())
{
	xModelRot = 0;
	yModelRot = 0;
	zModelRot = 0;

	xCamPos = 0;
	yCamPos = 0;
	zCamPos = 0;

	fovY = 0;

	this->setSurfaceType(OpenGLSurface);

	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setMajorVersion(3);
	format.setMinorVersion(3);
	format.setSamples(4);
	//format.setProfile(QSurfaceFormat::CoreProfile);
	format.setProfile(QSurfaceFormat::CompatibilityProfile);

	this->setFormat(format);
	create();

	mContext = new QOpenGLContext();
	mContext->setFormat(format);
	mContext->create();

	mScene->setContext(mContext);

	printContextInfos();
	initializeGl();

	connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeGl()));
	connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeGl()));
	resize(QSize(800, 600));

	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(updateScene()));
	timer->start(16);
}

CustomGLVAOWidget::~CustomGLVAOWidget()
{

}

void CustomGLVAOWidget::printContextInfos()
{
	if (!mContext->isValid())
		qDebug() << "\nThe OpenGL context is invalid!\n";

	mContext->makeCurrent(this);

	qDebug() << "\nWindow format version is: "
		<< format().majorVersion() << "."
		<< format().minorVersion();

	qDebug() << "Context format version is: "
		<< mContext->format().majorVersion()
		<< "." << mContext->format().minorVersion() << "\n";

	infoGL();
}

void CustomGLVAOWidget::initializeGl()
{
	mContext->makeCurrent(this);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	mScene->initialize();
	glCheckError();
}

void CustomGLVAOWidget::paintGl()
{
	if (!isExposed()) return;
	mContext->makeCurrent(this);
	mScene->render(xModelRot, yModelRot, zModelRot, xCamPos, yCamPos, zCamPos, fovY);
	mContext->swapBuffers(this);
	mContext->doneCurrent();
}

void CustomGLVAOWidget::resizeGl()
{
	mContext->makeCurrent(this);
	mScene->resize(width(), height());
}

void CustomGLVAOWidget::updateScene()
{
	mScene->update(0.0f);
	paintGl();
}

static void qNormalizeAngle(int& angle)
{
	angle = angle % 181;
}

void CustomGLVAOWidget::setXModelRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != xModelRot)
	{
		xModelRot = angle;
		emit xModelRotationChanged(angle);
		updateScene();
	}
}

void CustomGLVAOWidget::setYModelRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != yModelRot)
	{
		yModelRot = angle;
		emit yModelRotationChanged(angle);
		updateScene();
	}
}

void CustomGLVAOWidget::setZModelRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != zModelRot)
	{
		zModelRot = angle;
		emit zModelRotationChanged(angle);
		updateScene();
	}
}

void CustomGLVAOWidget::setXCamPosition(int pos)
{
	if (pos != xCamPos)
	{
		xCamPos = pos;
		emit xCamPositionChanged(pos);
		updateScene();
	}
}

void CustomGLVAOWidget::setYCamPosition(int pos)
{
	if (pos != yCamPos)
	{
		yCamPos = pos;
		emit yCamPositionChanged(pos);
		updateScene();
	}
}

void CustomGLVAOWidget::setZCamPosition(int pos)
{
	if (pos != zCamPos)
	{
		zCamPos = pos;
		emit zCamPositionChanged(pos);
		updateScene();
	}
}

void CustomGLVAOWidget::setFovY(int angle)
{
	if (angle != fovY)
	{
		fovY = angle;
		emit fovYChanged(angle);
		updateScene();
	}
}

void CustomGLVAOWidget::mousePressEvent(QMouseEvent *event)
{
	lastPos = event->pos();
}

void CustomGLVAOWidget::mouseMoveEvent(QMouseEvent *event)
{
	int dx = event->x() - lastPos.x();
	int dy = event->y() - lastPos.y();

	if (event->buttons() & Qt::LeftButton)
	{
		setXModelRotation(xModelRot + dy);
		setYModelRotation(yModelRot + dx);
	}
	else if (event->buttons() & Qt::RightButton)
	{
		setXModelRotation(xModelRot + dy);
		setZModelRotation(zModelRot + dx);
	}

	lastPos = event->pos();
}
