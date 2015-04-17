#ifndef CUSTOMGLWIDGET_H
#define CUSTOMGLWIDGET_H

#include <QGLWidget>
#include <QtWidgets>
#include <QtOpenGL>

class CustomGLWidget : public QGLWidget
{
	Q_OBJECT

public:
	explicit CustomGLWidget(QWidget *parent = 0);
	~CustomGLWidget();

private:
	void draw();

	int xRot;
	int yRot;
	int zRot;

	QPoint lastPos;

protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);

	QSize minimumSizeHint() const;
	QSize sizeHint() const;
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);

public slots: // slots for xyz-rotation slider 
	void setXRotation(int angle); 
	void setYRotation(int angle); 
	void setZRotation(int angle);

signals: // signaling rotation from mouse movement 
	void xRotationChanged(int angle);
	void yRotationChanged(int angle);
	void zRotationChanged(int angle);
};

#endif // CUSTOMGLWIDGET_H
