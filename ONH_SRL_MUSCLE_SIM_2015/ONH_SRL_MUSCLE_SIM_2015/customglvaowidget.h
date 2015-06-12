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
	
	void updateText();
	void loadModels();

private:
	void logSeparator(void);
	void appendToLog(QString text);
	void printContextInfos(void);
	void initializeGl(void);
	void infoGL(void);

	QOpenGLContext *mContext;
	QScopedPointer<AbstractScene> mScene;
	int xModelRot;
	int yModelRot;
	int zModelRot;
	int xCamPos;
	int yCamPos;
	int zCamPos;
	int fovY;
	int transpFactor;
	QPoint lastPos;
	QString logText;
	QDateTime dateTime;

	bool show_ANCONEUS;
	bool show_BRACHIALIS;
	bool show_BRACHIORDIALIS;
	bool show_PRONATOR_TERES;
	bool show_BICEPS_BRACHII;
	bool show_TRICEPS_BRACHII;
	bool show_OTHER;

protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void wheelEvent(QWheelEvent *event);

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
	void setTranspFactor(int factor);
	void resetView(void);

	void toggle_ANCONEUS(bool val);
	void toggle_BRACHIALIS(bool val);
	void toggle_BRACHIORDIALIS(bool val);
	void toggle_PRONATOR_TERES(bool val);
	void toggle_BICEPS_BRACHII(bool val);
	void toggle_TRICEPS_BRACHII(bool val);
	void toggle_OTHER(bool val);

signals:
	void xModelRotationChanged(int angle);
	void yModelRotationChanged(int angle);
	void zModelRotationChanged(int angle);

	void xCamPositionChanged(int pos);
	void yCamPositionChanged(int pos);
	void zCamPositionChanged(int pos);

	void fovYChanged(int angle);
	void transpFactorChanged(int factor);
	void logTextChanged(QString text);
};

#endif // CUSTOMGLVAOWIDGET_H
