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

	transpFactor = 50;

	show_ANCONEUS			= true;
	show_BRACHIALIS			= true;
	show_BRACHIORDIALIS		= true;
	show_PRONATOR_TERES		= true;
	show_BICEPS_BRACHII		= true;
	show_TRICEPS_BRACHII	= true;
	show_OTHER				= true;
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
	mScene->render(xModelRot, yModelRot, zModelRot, xCamPos, yCamPos, zCamPos, fovY, transpFactor);
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

void CustomGLVAOWidget::setFovY(int angle)
{
	if (angle != fovY)
	{
		fovY = angle;
		emit fovYChanged(angle);
		updateScene();
	}
}

void CustomGLVAOWidget::setTranspFactor(int factor)
{
	if (factor != transpFactor)
	{
		transpFactor = factor;
		emit transpFactorChanged(factor);
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

void CustomGLVAOWidget::toggle_BRACHIORDIALIS(bool val)
{
	if (val != show_BRACHIORDIALIS)
	{
		show_BRACHIORDIALIS = val;
		mScene->toggle_BRACHIORDIALIS(val);
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
	setXCamPosition(10);
	setYCamPosition(10);
	setZCamPosition(40);
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

void CustomGLVAOWidget::loadModels()
{
	appendToLog("Loading 'ANCONEUS'...");
	if (mScene->load_ANCONEUS() == 0)
	{
		appendToLog("Done!");
	}
	else
	{
		appendToLog("ERROR LOADING 'ANCONEUS'");
	}
	logSeparator();

	appendToLog("Loading 'BRACHIALIS'...");
	if (mScene->load_BRACHIALIS() == 0)
	{
		appendToLog("Done!");
	}
	else
	{
		appendToLog("ERROR LOADING 'BRACHIALIS'");
	}
	logSeparator();

	appendToLog("Loading 'BRACHIORDIALIS'...");
	if(mScene->load_BRACHIORDIALIS() == 0)
	{
		appendToLog("Done!");
	}
	else
	{
		appendToLog("ERROR LOADING 'BRACHIORDIALIS'");
	}
	logSeparator();
	
	appendToLog("Loading 'PRONATOR TERES'...");
	if(mScene->load_PRONATOR_TERES() == 0)
	{
		appendToLog("Done!");
	}
	else
	{
		appendToLog("ERROR LOADING 'PRONATOR TERES'");
	}
	logSeparator();

	appendToLog("Loading 'BICEPS BRACHII'...");
	if(mScene->load_BICEPS_BRACHII() == 0)
	{
		appendToLog("Done!");
	}
	else
	{
		appendToLog("ERROR LOADING 'BICEPS BRACHII'");
	}
	logSeparator();

	appendToLog("Loading 'TRICEPS BRACHII'...");
	if(mScene->load_TRICEPS_BRACHII() == 0)
	{
		appendToLog("Done!");
	}
	else
	{
		appendToLog("ERROR LOADING 'TRICEPS BRACHII'");
	}
	logSeparator();

	appendToLog("Loading 'OTHER'...");
	if(mScene->load_OTHER() == 0)
	{
		appendToLog("Done!");
	}
	else
	{
		appendToLog("ERROR LOADING 'OTHER'");
	}
	logSeparator();
}
