#include "customglwidget.h"

CustomGLWidget::CustomGLWidget(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	xRot = 0;
	yRot = 0;
	zRot = 0;
}

CustomGLWidget::~CustomGLWidget()
{

}

QSize CustomGLWidget::minimumSizeHint() const
{
	return QSize(50, 50);
}

QSize CustomGLWidget::sizeHint() const
{
	return QSize(400, 400);
}

static void qNormalizeAngle(int& angle)
{
	angle = angle % 181;
}

void CustomGLWidget::setXRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != xRot)
	{
		xRot = angle;
		emit xRotationChanged(angle);
		updateGL();
	}
}

void CustomGLWidget::setYRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != yRot)
	{
		yRot = angle;
		emit yRotationChanged(angle);
		updateGL();
	}
}

void CustomGLWidget::setZRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != zRot)
	{
		zRot = angle;
		emit zRotationChanged(angle);
		updateGL();
	}
}

void CustomGLWidget::initializeGL()
{
	qglClearColor(Qt::black);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	static GLfloat lightPosition[4] = { 0, 5, 10, 1.0 };
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
}

void CustomGLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -10.0);
	glRotatef(xRot, 1.0, 0.0, 0.0);
	glRotatef(yRot, 0.0, 1.0, 0.0);
	glRotatef(zRot, 0.0, 0.0, 1.0);
	draw();
}

void CustomGLWidget::resizeGL(int width, int height)
{
	int side = qMin(width, height);
	glViewport((width - side) / 2, (height - side) / 2, side, side);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-2, +2, -2, +2, 1.0, 15.0);
	glMatrixMode(GL_MODELVIEW);
}

void CustomGLWidget::mousePressEvent(QMouseEvent *event)
{
	lastPos = event->pos();
}

void CustomGLWidget::mouseMoveEvent(QMouseEvent *event)
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

void CustomGLWidget::draw()
{
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT);
	//glColor3f(0.12f, 0.12f, 0.12f);
	qglColor(Qt::gray);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	qglColor(Qt::blue);

	glBegin(GL_QUADS);
		glNormal3f(0, 0, -1);
		glVertex3f(-1, -1, 0);
		glVertex3f(-1, 1, 0);
		glVertex3f(1, 1, 0);
		glVertex3f(1, -1, 0);
	glEnd();

	glBegin(GL_TRIANGLES);
		glNormal3f(0, -1, 0.707);
		glVertex3f(-1, -1, 0);
		glVertex3f(1, -1, 0);
		glVertex3f(0, 0, 1.2);
	glEnd();
	
	glBegin(GL_TRIANGLES);
		glNormal3f(1, 0, 0.707);
		glVertex3f(1, -1, 0);
		glVertex3f(1, 1, 0);
		glVertex3f(0, 0, 1.2);
	glEnd();
	
	glBegin(GL_TRIANGLES);
		glNormal3f(0, 1, 0.707);
		glVertex3f(1, 1, 0);
		glVertex3f(-1, 1, 0);
		glVertex3f(0, 0, 1.2);
	glEnd();
	
	glBegin(GL_TRIANGLES);
		glNormal3f(-1, 0, 0.707);
		glVertex3f(-1, 1, 0);
		glVertex3f(-1, -1, 0);
		glVertex3f(0, 0, 1.2);
	glEnd();
}
