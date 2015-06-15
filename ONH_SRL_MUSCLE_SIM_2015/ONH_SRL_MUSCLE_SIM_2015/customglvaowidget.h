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
#include <QThread>

class QOpenGLContext;

class ModelLoaderThread : public QThread
{
	Q_OBJECT

public:
	ModelLoaderThread(QSharedPointer<AbstractScene> _mScene, ARM_PART _part) : mScene(_mScene), part(_part)
	{

	};

protected:
	void run() Q_DECL_OVERRIDE
	{
		int result = -1;
		switch (part)
		{
		case AP_ANCONEUS:
			result = mScene->load_ANCONEUS();
			break;
		case AP_BRACHIALIS:
			result = mScene->load_BRACHIALIS();
			break;
		case AP_BRACHIORADIALIS:
			result = mScene->load_BRACHIORADIALIS();
			break;
		case AP_PRONATOR_TERES:
			result = mScene->load_PRONATOR_TERES();
			break;
		case AP_BICEPS_BRACHII:
			result = mScene->load_BICEPS_BRACHII();
			break;
		case AP_TRICEPS_BRACHII:
			result = mScene->load_TRICEPS_BRACHII();
			break;
		case AP_OTHER:
			result = mScene->load_OTHER();
			break;
		case AP_BONES:
			result = mScene->load_BONES();
			break;
		}
		emit modelLoaded(result);
	};
signals:
	void modelLoaded(const int result);
private:
	QSharedPointer<AbstractScene> mScene;
	ARM_PART part;
};

class CustomGLVAOWidget : public QWindow
{
	Q_OBJECT

public:
	explicit CustomGLVAOWidget(QScreen *screen = 0);
	~CustomGLVAOWidget();
	
	void updateText();
	void loadModels();
	void doResize(int width, int height);

private:
	void logSeparator(void);
	void appendToLog(QString text);
	void printContextInfos(void);
	void initializeGl(void);
	void infoGL(void);

	void handle_ANCONEUS_Result(int result);
	void handle_BRACHIALIS_Result(int result);
	void handle_BRACHIORADIALIS_Result(int result);
	void handle_PRONATOR_TERES_Result(int result);
	void handle_BICEPS_BRACHII_Result(int result);
	void handle_TRICEPS_BRACHII_Result(int result);
	void handle_OTHER_Result(int result);
	void handle_BONES_Result(int result);

	QTimer *timer;
	QOpenGLContext *mContext;
	QSharedPointer<AbstractScene> mScene;
	int xModelRot;
	int yModelRot;
	int zModelRot;
	int xCamPos;
	int yCamPos;
	int zCamPos;
	int xCamPiv;
	int yCamPiv;
	int zCamPiv;
	int fovY;

	QPoint lastPos;
	QString logText;
	QDateTime dateTime;

	bool show_ANCONEUS;
	bool show_BRACHIALIS;
	bool show_BRACHIORADIALIS;
	bool show_PRONATOR_TERES;
	bool show_BICEPS_BRACHII;
	bool show_TRICEPS_BRACHII;
	bool show_OTHER;
	bool show_BONES;

	bool show_GRID;
	bool show_AXES;

	bool wireframe_ANCONEUS;
	bool wireframe_BRACHIALIS;
	bool wireframe_BRACHIORADIALIS;
	bool wireframe_PRONATOR_TERES;
	bool wireframe_BICEPS_BRACHII;
	bool wireframe_TRICEPS_BRACHII;
	bool wireframe_OTHER;
	bool wireframe_BONES;

	int opacity_ANCONEUS;
	int opacity_BRACHIALIS;
	int opacity_BRACHIORADIALIS;
	int opacity_PRONATOR_TERES;
	int opacity_BICEPS_BRACHII;
	int opacity_TRICEPS_BRACHII;
	int opacity_OTHER;
	int opacity_BONES;


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

	void setXCamPivot(int pos);
	void setYCamPivot(int pos);
	void setZCamPivot(int pos);

	void setFovY(int angle);
	void resetView(void);

	void toggle_ANCONEUS(bool val);
	void toggle_BRACHIALIS(bool val);
	void toggle_BRACHIORADIALIS(bool val);
	void toggle_PRONATOR_TERES(bool val);
	void toggle_BICEPS_BRACHII(bool val);
	void toggle_TRICEPS_BRACHII(bool val);
	void toggle_OTHER(bool val);
	void toggle_BONES(bool val);

	void toggle_GRID(bool val);
	void toggle_AXES(bool val);

	void toggle_ANCONEUS_wireframe(bool val);
	void toggle_BRACHIALIS_wireframe(bool val);
	void toggle_BRACHIORADIALIS_wireframe(bool val);
	void toggle_PRONATOR_TERES_wireframe(bool val);
	void toggle_BICEPS_BRACHII_wireframe(bool val);
	void toggle_TRICEPS_BRACHII_wireframe(bool val);
	void toggle_OTHER_wireframe(bool val);
	void toggle_BONES_wireframe(bool val);

	void set_ANCONEUS_opacity(int val);
	void set_BRACHIALIS_opacity(int val);
	void set_BRACHIORADIALIS_opacity(int val);
	void set_PRONATOR_TERES_opacity(int val);
	void set_BICEPS_BRACHII_opacity(int val);
	void set_TRICEPS_BRACHII_opacity(int val);
	void set_OTHER_opacity(int val);
	void set_BONES_opacity(int val);

signals:
	void xModelRotationChanged(int angle);
	void yModelRotationChanged(int angle);
	void zModelRotationChanged(int angle);

	void xCamPositionChanged(int pos);
	void yCamPositionChanged(int pos);
	void zCamPositionChanged(int pos);

	void xCamPivotChanged(int pos);
	void yCamPivotChanged(int pos);
	void zCamPivotChanged(int pos);

	void fovYChanged(int angle);
	void logTextChanged(QString text);
};

#endif // CUSTOMGLVAOWIDGET_H
