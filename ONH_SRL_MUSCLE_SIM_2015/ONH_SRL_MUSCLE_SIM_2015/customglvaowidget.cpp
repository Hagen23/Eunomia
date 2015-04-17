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
	xRot = 0;
	yRot = 0;
	zRot = 0;

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
	resize(QSize(320, 240));

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
	mScene->initialize();
	glCheckError();
}

void CustomGLVAOWidget::paintGl()
{
	if (!isExposed()) return;
	mContext->makeCurrent(this);
	mScene->render(xRot, yRot, zRot);
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

void CustomGLVAOWidget::setXRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != xRot)
	{
		xRot = angle;
		emit xRotationChanged(angle);
		updateScene();
	}
}

void CustomGLVAOWidget::setYRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != yRot)
	{
		yRot = angle;
		emit yRotationChanged(angle);
		updateScene();
	}
}

void CustomGLVAOWidget::setZRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != zRot)
	{
		zRot = angle;
		emit zRotationChanged(angle);
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
		setXRotation(xRot + dy);
		setYRotation(yRot + dx);
	}
	else if (event->buttons() & Qt::RightButton)
	{
		setXRotation(xRot + dy);
		setZRotation(zRot + dx);
	}

	lastPos = event->pos();
}
