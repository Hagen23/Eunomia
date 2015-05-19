#include "onh_srl_muscle_sim_2015.h"

ONH_SRL_MUSCLE_SIM_2015::ONH_SRL_MUSCLE_SIM_2015(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	vaoWidget = new CustomGLVAOWidget();
	QWidget *container = QWidget::createWindowContainer(vaoWidget, ui.centralWidget);
	container->setFocusPolicy(Qt::TabFocus);
	container->setGeometry(370, 30, 800, 600);

	connect(vaoWidget, SIGNAL(xModelRotationChanged(int)), ui.xModelRotSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(yModelRotationChanged(int)), ui.yModelRotSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(zModelRotationChanged(int)), ui.zModelRotSlider, SLOT(setValue(int)));

	connect(ui.xModelRotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setXModelRotation(int)));
	connect(ui.yModelRotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setYModelRotation(int)));
	connect(ui.zModelRotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setZModelRotation(int)));

	connect(vaoWidget, SIGNAL(xCamPositionChanged(int)), ui.xCamPosSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(yCamPositionChanged(int)), ui.yCamPosSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(zCamPositionChanged(int)), ui.zCamPosSlider, SLOT(setValue(int)));

	connect(ui.xCamPosSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setXCamPosition(int)));
	connect(ui.yCamPosSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setYCamPosition(int)));
	connect(ui.zCamPosSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setZCamPosition(int)));

	connect(vaoWidget, SIGNAL(fovYChanged(int)), ui.fovyDial, SLOT(setValue(int)));
	connect(ui.fovyDial, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setFovY(int)));

	connect(vaoWidget, SIGNAL(transpFactorChanged(int)), ui.modelTranspSlider, SLOT(setValue(int)));
	connect(ui.modelTranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setTranspFactor(int)));

	connect(vaoWidget, SIGNAL(logTextChanged(QString)), ui.logPlainTextEdit, SLOT(setPlainText(QString)));

	vaoWidget->resetView();
	vaoWidget->updateText();
}

ONH_SRL_MUSCLE_SIM_2015::~ONH_SRL_MUSCLE_SIM_2015()
{

}

void ONH_SRL_MUSCLE_SIM_2015::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
		close();
	else
		QWidget::keyPressEvent(e);
}
