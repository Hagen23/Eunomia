#include "onh_srl_muscle_sim_2015.h"

ONH_SRL_MUSCLE_SIM_2015::ONH_SRL_MUSCLE_SIM_2015(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	vaoWidget = new CustomGLVAOWidget();
	container = QWidget::createWindowContainer(vaoWidget, ui.centralWidget);
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

	connect(vaoWidget, SIGNAL(xCamPivotChanged(int)), ui.xCamPivotSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(yCamPivotChanged(int)), ui.yCamPivotSlider, SLOT(setValue(int)));
	connect(vaoWidget, SIGNAL(zCamPivotChanged(int)), ui.zCamPivotSlider, SLOT(setValue(int)));

	connect(ui.xCamPosSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setXCamPosition(int)));
	connect(ui.yCamPosSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setYCamPosition(int)));
	connect(ui.zCamPosSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setZCamPosition(int)));

	connect(ui.xCamPivotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setXCamPivot(int)));
	connect(ui.yCamPivotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setYCamPivot(int)));
	connect(ui.zCamPivotSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setZCamPivot(int)));

	connect(vaoWidget, SIGNAL(fovYChanged(int)), ui.fovyDial, SLOT(setValue(int)));
	connect(ui.fovyDial, SIGNAL(valueChanged(int)), vaoWidget, SLOT(setFovY(int)));

	connect(ui.ANCONEUS_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_ANCONEUS_opacity(int)));
	connect(ui.BRACHIALIS_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_BRACHIALIS_opacity(int)));
	connect(ui.BRACHIORADIALIS_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_BRACHIORADIALIS_opacity(int)));
	connect(ui.PRONATOR_TERES_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_PRONATOR_TERES_opacity(int)));
	connect(ui.BICEPS_BRACHII_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_BICEPS_BRACHII_opacity(int)));
	connect(ui.TRICEPS_BRACHII_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_TRICEPS_BRACHII_opacity(int)));
	connect(ui.OTHER_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_OTHER_opacity(int)));
	connect(ui.BONES_TranspSlider, SIGNAL(valueChanged(int)), vaoWidget, SLOT(set_BONES_opacity(int)));

	connect(ui.resetView_button, SIGNAL(clicked()), vaoWidget, SLOT(resetView()));

	connect(ui.ANCONEUS_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_ANCONEUS(bool)));
	connect(ui.BRACHIALIS_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BRACHIALIS(bool)));
	connect(ui.BRACHIORADIALIS_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BRACHIORADIALIS(bool)));
	connect(ui.PRONATOR_TERES_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_PRONATOR_TERES(bool)));
	connect(ui.BICEPS_BRACHII_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BICEPS_BRACHII(bool)));
	connect(ui.TRICEPS_BRACHII_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_TRICEPS_BRACHII(bool)));
	connect(ui.OTHER_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_OTHER(bool)));
	connect(ui.BONES_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BONES(bool)));
	connect(ui.GRID_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_GRID(bool)));
	connect(ui.AXES_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_AXES(bool)));

	connect(ui.ANCONEUS_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_ANCONEUS_wireframe(bool)));
	connect(ui.BRACHIALIS_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BRACHIALIS_wireframe(bool)));
	connect(ui.BRACHIORADIALIS_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BRACHIORADIALIS_wireframe(bool)));
	connect(ui.PRONATOR_TERES_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_PRONATOR_TERES_wireframe(bool)));
	connect(ui.BICEPS_BRACHII_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BICEPS_BRACHII_wireframe(bool)));
	connect(ui.TRICEPS_BRACHII_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_TRICEPS_BRACHII_wireframe(bool)));
	connect(ui.OTHER_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_OTHER_wireframe(bool)));
	connect(ui.BONES_wireframe_checkBox, SIGNAL(toggled(bool)), vaoWidget, SLOT(toggle_BONES_wireframe(bool)));

	connect(vaoWidget, SIGNAL(logTextChanged(QString)), ui.logPlainTextEdit, SLOT(setPlainText(QString)));

	vaoWidget->resetView();
	vaoWidget->updateText();
}

ONH_SRL_MUSCLE_SIM_2015::~ONH_SRL_MUSCLE_SIM_2015()
{

}

void ONH_SRL_MUSCLE_SIM_2015::loadModels(void)
{
	vaoWidget->loadModels();
}

void ONH_SRL_MUSCLE_SIM_2015::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
		close();
	else
		QWidget::keyPressEvent(e);
}

void ONH_SRL_MUSCLE_SIM_2015::resizeEvent(QResizeEvent* event)
{
	QMainWindow::resizeEvent(event);
	if (this)
	{
		qDebug() << "NEW WINDOW SIZE: " << this->width() << " X " << this->height();
		int glWidth = this->width() - 400;
		int glHeight = (glWidth * 3) / 4;		//TRY 4:3
		if (glHeight > (this->height() - 99))
		{
			glHeight = (glWidth * 9) / 16;		//BETTER 16:9
		}
		qDebug() << "NEW GL SIZE: " << glWidth << " X " << glHeight;
		container->setGeometry(370, 30, glWidth, glHeight);
		vaoWidget->doResize(glWidth, glHeight);
	}
}
