#include "onh_srl_muscle_sim_2015.h"

ONH_SRL_MUSCLE_SIM_2015::ONH_SRL_MUSCLE_SIM_2015(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	vaoWidget = new CustomGLVAOWidget();
	QWidget *container = QWidget::createWindowContainer(vaoWidget, ui.centralWidget);
	container->setFocusPolicy(Qt::TabFocus);
	container->setGeometry(270, 40, 320, 240);

	connect(vaoWidget, SIGNAL(xRotationChanged(int)), ui.xRotSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(yRotationChanged(int)), ui.yRotSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(zRotationChanged(int)), ui.zRotSlider, SLOT(setValue(int)));

	connect(ui.xRotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setXRotation(int)));
	connect(ui.yRotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setYRotation(int)));
	connect(ui.zRotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setZRotation(int)));
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
