#include "customglvaowidget.h"

CustomGLVAOWidget::CustomGLVAOWidget(QScreen *screen) : QWindow(screen), mScene(new QtOpenGLScene())
{
	logText = QString("");
	appendToLog("ONH - SRL MUSCLE SIM 2015");
	logSeparator();

	xModelRot = 0;
	yModelRot = 0;
	zModelRot = 0;

	xCamPos = 0;
	yCamPos = 0;
	zCamPos = 0;

	xCamPiv = 0;
	yCamPiv = 0;
	zCamPiv = 0;

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

	//connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeGl()));
	//connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeGl()));
	//resize(QSize(800, 600));
	resize(width(), height());

	timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(updateScene()));
	timer->start(16);

	show_ANCONEUS				= true;
	show_BRACHIALIS				= true;
	show_BRACHIORADIALIS		= true;
	show_PRONATOR_TERES			= true;
	show_BICEPS_BRACHII			= true;
	show_TRICEPS_BRACHII		= true;
	show_OTHER					= true;
	show_BONES					= true;

	show_GRID					= true;
	show_AXES					= true;

	wireframe_ANCONEUS			= false;
	wireframe_BRACHIALIS		= false;
	wireframe_BRACHIORADIALIS	= false;
	wireframe_PRONATOR_TERES	= false;
	wireframe_BICEPS_BRACHII	= false;
	wireframe_TRICEPS_BRACHII	= false;
	wireframe_OTHER				= false;
	wireframe_BONES				= false;

	opacity_ANCONEUS			= 100;
	opacity_BRACHIALIS			= 100;
	opacity_BRACHIORADIALIS		= 100;
	opacity_PRONATOR_TERES		= 100;
	opacity_BICEPS_BRACHII		= 100;
	opacity_TRICEPS_BRACHII		= 100;
	opacity_OTHER				= 100;
	opacity_BONES				= 100;
}

CustomGLVAOWidget::~CustomGLVAOWidget()
{

}

void CustomGLVAOWidget::infoGL()
{
	glCheckError();
	const char *str;
	qDebug() << "\nOpenGL info with GL functions:";
	//appendToLog("OpenGL info with GL functions:");
	str = (const char*)glGetString(GL_RENDERER);
	qDebug() << "Renderer : " << QString(str);
	appendToLog("Renderer : " + QString(str));
	str = (const char*)glGetString(GL_VENDOR);
	qDebug() << "Vendor : " << QString(str);
	appendToLog("Vendor : " + QString(str));
	str = (const char*)glGetString(GL_VERSION);
	qDebug() << "OpenGL Version : " << QString(str);
	appendToLog("OpenGL Version : " + QString(str));
	str = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
	qDebug() << "GLSL Version : " << QString(str);
	appendToLog("GLSL Version : " + QString(str));
	glCheckError();
}

void CustomGLVAOWidget::printContextInfos()
{
	if (!mContext->isValid())
	{
		qDebug() << "\nThe OpenGL context is invalid!\n";
		appendToLog("The OpenGL context is invalid!");
	}
	mContext->makeCurrent(this);

	QString fmt = "Window format version: " + QString::number(format().majorVersion()) + "." + QString::number(format().minorVersion());
	qDebug() << "\nWindow format version is: " << format().majorVersion() << "." << format().minorVersion();
	appendToLog(fmt);
	QString ctx = "Context format version: " + QString::number(mContext->format().majorVersion()) + "." + QString::number(mContext->format().minorVersion());
	qDebug() << "Context format version is: " << mContext->format().majorVersion() << "." << mContext->format().minorVersion() << "\n";
	appendToLog(ctx);
	infoGL();
	logSeparator();
}

void CustomGLVAOWidget::initializeGl()
{
	mContext->makeCurrent(this);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_CULL_FACE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	mScene->initialize();
	glCheckError();
	resetView();
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
	timer->stop();
	mScene->resize(width(), height());
	timer->start(16);
}

void CustomGLVAOWidget::updateScene()
{
	mScene->update(0.0f);
	paintGl();
}

static void qNormalizeAngle(int& angle)
{
	if (angle < -180) angle = 180;
	else if (angle > 180) angle = -180;
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

void CustomGLVAOWidget::setXCamPivot(int pos)
{
	if (pos != xCamPiv)
	{
		xCamPiv = pos;
		mScene->setXPivot((float)pos / 20.0f);
		emit xCamPivotChanged(pos);
		updateScene();
	}
}

void CustomGLVAOWidget::setYCamPivot(int pos)
{
	if (pos != yCamPiv)
	{
		yCamPiv = pos;
		mScene->setYPivot((float)pos / 20.0f);
		emit yCamPivotChanged(pos);
		updateScene();
	}
}

void CustomGLVAOWidget::setZCamPivot(int pos)
{
	if (pos != zCamPiv)
	{
		zCamPiv = pos;
		mScene->setZPivot((float)pos / 20.0f);
		emit zCamPivotChanged(pos);
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

void CustomGLVAOWidget::toggle_ANCONEUS(bool val)
{
	if (val != show_ANCONEUS)
	{
		show_ANCONEUS = val;
		mScene->toggle_ANCONEUS(val);
	}
}

void CustomGLVAOWidget::toggle_BRACHIALIS(bool val)
{
	if (val != show_BRACHIALIS)
	{
		show_BRACHIALIS = val;
		mScene->toggle_BRACHIALIS(val);
	}
}

void CustomGLVAOWidget::toggle_BRACHIORADIALIS(bool val)
{
	if (val != show_BRACHIORADIALIS)
	{
		show_BRACHIORADIALIS = val;
		mScene->toggle_BRACHIORADIALIS(val);
	}
}

void CustomGLVAOWidget::toggle_PRONATOR_TERES(bool val)
{
	if (val != show_PRONATOR_TERES)
	{
		show_PRONATOR_TERES = val;
		mScene->toggle_PRONATOR_TERES(val);
	}
}

void CustomGLVAOWidget::toggle_BICEPS_BRACHII(bool val)
{
	if (val != show_BICEPS_BRACHII)
	{
		show_BICEPS_BRACHII = val;
		mScene->toggle_BICEPS_BRACHII(val);
	}
}

void CustomGLVAOWidget::toggle_TRICEPS_BRACHII(bool val)
{
	if (val != show_TRICEPS_BRACHII)
	{
		show_TRICEPS_BRACHII = val;
		mScene->toggle_TRICEPS_BRACHII(val);
	}
}

void CustomGLVAOWidget::toggle_OTHER(bool val)
{
	if (val != show_OTHER)
	{
		show_OTHER = val;
		mScene->toggle_OTHER(val);
	}
}

void CustomGLVAOWidget::toggle_BONES(bool val)
{
	if (val != show_BONES)
	{
		show_BONES = val;
		mScene->toggle_BONES(val);
	}
}

void CustomGLVAOWidget::toggle_GRID(bool val)
{
	if (val != show_GRID)
	{
		show_GRID = val;
		mScene->toggle_GRID(val);
	}
}

void CustomGLVAOWidget::toggle_AXES(bool val)
{
	if (val != show_AXES)
	{
		show_AXES = val;
		mScene->toggle_AXES(val);
	}
}

void CustomGLVAOWidget::toggle_ANCONEUS_wireframe(bool val)
{
	if (val != wireframe_ANCONEUS)
	{
		wireframe_ANCONEUS = val;
		mScene->toggle_ANCONEUS_wireframe(val);
	}
}

void CustomGLVAOWidget::toggle_BRACHIALIS_wireframe(bool val)
{
	if (val != wireframe_BRACHIALIS)
	{
		wireframe_BRACHIALIS = val;
		mScene->toggle_BRACHIALIS_wireframe(val);
	}
}

void CustomGLVAOWidget::toggle_BRACHIORADIALIS_wireframe(bool val)
{
	if (val != wireframe_BRACHIORADIALIS)
	{
		wireframe_BRACHIORADIALIS = val;
		mScene->toggle_BRACHIORADIALIS_wireframe(val);
	}
}

void CustomGLVAOWidget::toggle_PRONATOR_TERES_wireframe(bool val)
{
	if (val != wireframe_PRONATOR_TERES)
	{
		wireframe_PRONATOR_TERES = val;
		mScene->toggle_PRONATOR_TERES_wireframe(val);
	}
}

void CustomGLVAOWidget::toggle_BICEPS_BRACHII_wireframe(bool val)
{
	if (val != wireframe_BICEPS_BRACHII)
	{
		wireframe_BICEPS_BRACHII = val;
		mScene->toggle_BICEPS_BRACHII_wireframe(val);
	}
}

void CustomGLVAOWidget::toggle_TRICEPS_BRACHII_wireframe(bool val)
{
	if (val != wireframe_TRICEPS_BRACHII)
	{
		wireframe_TRICEPS_BRACHII = val;
		mScene->toggle_TRICEPS_BRACHII_wireframe(val);
	}
}

void CustomGLVAOWidget::toggle_OTHER_wireframe(bool val)
{
	if (val != wireframe_OTHER)
	{
		wireframe_OTHER = val;
		mScene->toggle_OTHER_wireframe(val);
	}
}

void CustomGLVAOWidget::toggle_BONES_wireframe(bool val)
{
	if (val != wireframe_BONES)
	{
		wireframe_BONES = val;
		mScene->toggle_BONES_wireframe(val);
	}
}

void CustomGLVAOWidget::set_ANCONEUS_opacity(int val)
{
	if (val != opacity_ANCONEUS && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_ANCONEUS_opacity(o);
		opacity_ANCONEUS = val;
	}
}

void CustomGLVAOWidget::set_BRACHIALIS_opacity(int val)
{
	if (val != opacity_BRACHIALIS && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_BRACHIALIS_opacity(o);
		opacity_BRACHIALIS = val;
	}
}

void CustomGLVAOWidget::set_BRACHIORADIALIS_opacity(int val)
{
	if (val != opacity_BRACHIORADIALIS && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_BRACHIORADIALIS_opacity(o);
		opacity_BRACHIORADIALIS = val;
	}
}

void CustomGLVAOWidget::set_PRONATOR_TERES_opacity(int val)
{
	if (val != opacity_PRONATOR_TERES && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_PRONATOR_TERES_opacity(o);
		opacity_PRONATOR_TERES = val;
	}
}

void CustomGLVAOWidget::set_BICEPS_BRACHII_opacity(int val)
{
	if (val != opacity_BICEPS_BRACHII && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_BICEPS_BRACHII_opacity(o);
		opacity_BICEPS_BRACHII = val;
	}
}

void CustomGLVAOWidget::set_TRICEPS_BRACHII_opacity(int val)
{
	if (val != opacity_TRICEPS_BRACHII && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_TRICEPS_BRACHII_opacity(o);
		opacity_TRICEPS_BRACHII = val;
	}
}

void CustomGLVAOWidget::set_OTHER_opacity(int val)
{
	if (val != opacity_OTHER && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_OTHER_opacity(o);
		opacity_OTHER = val;
	}
}

void CustomGLVAOWidget::set_BONES_opacity(int val)
{
	if (val != opacity_BONES && val > 0)
	{
		float o = ((float)val) / 100.0f;
		mScene->set_BONES_opacity(o);
		opacity_BONES = val;
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
	else if (event->buttons() & Qt::MiddleButton)
	{
		int deltaX = (dx < 0) ? -1 : ((dx > 0) ? 1 : 0);
		setXCamPosition(xCamPos + deltaX);
		setXCamPivot(xCamPiv + deltaX);

		int deltaY = (dy < 0) ? -1 : ((dy > 0) ? 1 : 0);
		setYCamPosition(yCamPos - deltaY);
		setYCamPivot(yCamPiv - deltaY);
	}

	lastPos = event->pos();
}

void CustomGLVAOWidget::wheelEvent(QWheelEvent *event)
{
	QPoint numPixels = event->pixelDelta();
	QPoint numDegrees = event->angleDelta() / 8;

	if (!numDegrees.isNull())
	{
		int dy = numDegrees.y();
		int numSteps = fovY + dy / 15;
		if (numSteps >= -15 && numSteps <= 17)
		{
			setFovY(numSteps);
		}
	}
	event->accept();
}

void CustomGLVAOWidget::resetView()
{
	setXModelRotation(-180);
	setYModelRotation(0);
	setZModelRotation(0);
	setXCamPosition(30);
	setYCamPosition(15);
	setZCamPosition(30);
	setXCamPivot(0);
	setYCamPivot(0);
	setZCamPivot(0);
	setFovY(0);
}

void CustomGLVAOWidget::appendToLog(QString text)
{
	dateTime = QDateTime::currentDateTime();
	logText.append(dateTime.toString("yyyyMMdd_hhmmss_zzz"));
	logText.append(" ");
	logText.append(text);
	logText.append("\n");
	emit logTextChanged(logText);
}

void CustomGLVAOWidget::logSeparator(void)
{
	logText.append("--------------------\n");
	emit logTextChanged(logText);
}

void CustomGLVAOWidget::updateText()
{
	emit logTextChanged(logText);
}

void CustomGLVAOWidget::handle_ANCONEUS_Result(int result)
{
	QString bName = "ANCONEUS";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_ANCONEUS() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::handle_BRACHIALIS_Result(int result)
{
	QString bName = "BRACHIALIS";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_BRACHIALIS() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::handle_BRACHIORADIALIS_Result(int result)
{
	QString bName = "BRACHIORADIALIS";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_BRACHIORADIALIS() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::handle_PRONATOR_TERES_Result(int result)
{
	QString bName = "PRONATOR TERES";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_PRONATOR_TERES() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::handle_BICEPS_BRACHII_Result(int result)
{
	QString bName = "BICEPS BRACHII";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_BICEPS_BRACHII() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::handle_TRICEPS_BRACHII_Result(int result)
{
	QString bName = "TRICEPS BRACHII";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_TRICEPS_BRACHII() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::handle_OTHER_Result(int result)
{
	QString bName = "OTHER";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_OTHER() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::handle_BONES_Result(int result)
{
	QString bName = "BONES";
	if (result == 0)
	{
		appendToLog(QString("Loading '%1' OK").arg(bName));
		appendToLog(QString("Generating VAOs for '%1'...").arg(bName));
		mContext->makeCurrent(this);
		if (mScene->pack_BONES() == 0)
		{
			appendToLog(QString("'%1' VAOs generation OK").arg(bName));
		}
		else
		{
			appendToLog(QString("'%1' VAOs generation FAILED!").arg(bName));
		}
		mContext->doneCurrent();
	}
	else
	{
		appendToLog(QString("Loading '%1' FAILED!").arg(bName));
	}
}

void CustomGLVAOWidget::loadModels()
{
	appendToLog("Loading 'ANCONEUS'...");
	ModelLoaderThread* lm_ANCONEUS_Thread = new ModelLoaderThread(mScene, AP_ANCONEUS);
	connect(lm_ANCONEUS_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_ANCONEUS_Result);
	connect(lm_ANCONEUS_Thread, &ModelLoaderThread::finished, lm_ANCONEUS_Thread, &QObject::deleteLater);
	lm_ANCONEUS_Thread->start();

	appendToLog("Loading 'BRACHIALIS'...");
	ModelLoaderThread* lm_BRACHIALIS_Thread = new ModelLoaderThread(mScene, AP_BRACHIALIS);
	connect(lm_BRACHIALIS_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_BRACHIALIS_Result);
	connect(lm_BRACHIALIS_Thread, &ModelLoaderThread::finished, lm_BRACHIALIS_Thread, &QObject::deleteLater);
	lm_BRACHIALIS_Thread->start();

	appendToLog("Loading 'BRACHIORADIALIS'...");
	ModelLoaderThread* lm_BRACHIORADIALIS_Thread = new ModelLoaderThread(mScene, AP_BRACHIORADIALIS);
	connect(lm_BRACHIORADIALIS_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_BRACHIORADIALIS_Result);
	connect(lm_BRACHIORADIALIS_Thread, &ModelLoaderThread::finished, lm_BRACHIORADIALIS_Thread, &QObject::deleteLater);
	lm_BRACHIORADIALIS_Thread->start();

	appendToLog("Loading 'PRONATOR TERES'...");
	ModelLoaderThread* lm_PRONATOR_TERES_Thread = new ModelLoaderThread(mScene, AP_PRONATOR_TERES);
	connect(lm_PRONATOR_TERES_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_PRONATOR_TERES_Result);
	connect(lm_PRONATOR_TERES_Thread, &ModelLoaderThread::finished, lm_PRONATOR_TERES_Thread, &QObject::deleteLater);
	lm_PRONATOR_TERES_Thread->start();

	appendToLog("Loading 'BICEPS BRACHII'...");
	ModelLoaderThread* lm_BICEPS_BRACHII_Thread = new ModelLoaderThread(mScene, AP_BICEPS_BRACHII);
	connect(lm_BICEPS_BRACHII_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_BICEPS_BRACHII_Result);
	connect(lm_BICEPS_BRACHII_Thread, &ModelLoaderThread::finished, lm_BICEPS_BRACHII_Thread, &QObject::deleteLater);
	lm_BICEPS_BRACHII_Thread->start();

	appendToLog("Loading 'TRICEPS BRACHII'...");
	ModelLoaderThread* lm_TRICEPS_BRACHII_Thread = new ModelLoaderThread(mScene, AP_TRICEPS_BRACHII);
	connect(lm_TRICEPS_BRACHII_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_TRICEPS_BRACHII_Result);
	connect(lm_TRICEPS_BRACHII_Thread, &ModelLoaderThread::finished, lm_TRICEPS_BRACHII_Thread, &QObject::deleteLater);
	lm_TRICEPS_BRACHII_Thread->start();

	appendToLog("Loading 'OTHER'...");
	ModelLoaderThread* lm_OTHER_Thread = new ModelLoaderThread(mScene, AP_OTHER);
	connect(lm_OTHER_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_OTHER_Result);
	connect(lm_OTHER_Thread, &ModelLoaderThread::finished, lm_OTHER_Thread, &QObject::deleteLater);
	lm_OTHER_Thread->start();

	appendToLog("Loading 'BONES'...");
	ModelLoaderThread* lm_BONES_Thread = new ModelLoaderThread(mScene, AP_BONES);
	connect(lm_BONES_Thread, &ModelLoaderThread::modelLoaded, this, &CustomGLVAOWidget::handle_BONES_Result);
	connect(lm_BONES_Thread, &ModelLoaderThread::finished, lm_BONES_Thread, &QObject::deleteLater);
	lm_BONES_Thread->start();
}

void CustomGLVAOWidget::doResize(int width, int height)
{
	resize(QSize(width, height));
	mContext->makeCurrent(this);
	timer->stop();
	mScene->resize(width, height);
	timer->start(16);
}
