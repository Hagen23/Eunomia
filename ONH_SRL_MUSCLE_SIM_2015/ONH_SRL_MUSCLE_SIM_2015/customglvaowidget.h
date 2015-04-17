#ifndef CUSTOMGLVAOWIDGET_H
#define CUSTOMGLVAOWIDGET_H

#include <QWindow>
#include <QtWidgets>
#include <qtimer.h>
#include "abstractscene.h"
#include "qtopenglscene.h"
#include "glassert.h"

#include <iostream>
#include <QOpenGLContext>

class QOpenGLContext;

class CustomGLVAOWidget : public QWindow
{
	Q_OBJECT

public:
	explicit CustomGLVAOWidget(QScreen *screen = 0);
	~CustomGLVAOWidget();

private:
	void printContextInfos();
	void initializeGl();

	QOpenGLContext *mContext;
	QScopedPointer<AbstractScene> mScene;
	int xRot;
	int yRot;
	int zRot;
	QPoint lastPos;

protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);

protected slots :
	void resizeGl();
	void paintGl();
	void updateScene();

public slots: // slots for xyz-rotation slider 
	void setXRotation(int angle);
	void setYRotation(int angle);
	void setZRotation(int angle);

signals: // signaling rotation from mouse movement 
	void xRotationChanged(int angle);
	void yRotationChanged(int angle);
	void zRotationChanged(int angle);
};

#endif // CUSTOMGLVAOWIDGET_H
