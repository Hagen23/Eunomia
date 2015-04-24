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
	int xModelRot;
	int yModelRot;
	int zModelRot;
	int xCamPos;
	int yCamPos;
	int zCamPos;
	int fovY;
	QPoint lastPos;

protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);

protected slots :
	void resizeGl();
	void paintGl();
	void updateScene();

public slots:
	void setXModelRotation(int angle);
	void setYModelRotation(int angle);
	void setZModelRotation(int angle);

	void setXCamPosition(int pos);
	void setYCamPosition(int pos);
	void setZCamPosition(int pos);

	void setFovY(int angle);

signals:
	void xModelRotationChanged(int angle);
	void yModelRotationChanged(int angle);
	void zModelRotationChanged(int angle);

	void xCamPositionChanged(int pos);
	void yCamPositionChanged(int pos);
	void zCamPositionChanged(int pos);

	void fovYChanged(int angle);
};

#endif // CUSTOMGLVAOWIDGET_H
