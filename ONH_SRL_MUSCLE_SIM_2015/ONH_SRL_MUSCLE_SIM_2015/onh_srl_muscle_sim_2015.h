#ifndef ONH_SRL_MUSCLE_SIM_2015_H
#define ONH_SRL_MUSCLE_SIM_2015_H

#include <QtWidgets/QMainWindow>
#include <QWidget>
#include <QSlider>
#include "ui_onh_srl_muscle_sim_2015.h"
#include "customglvaowidget.h"

class ONH_SRL_MUSCLE_SIM_2015 : public QMainWindow
{
	Q_OBJECT

public:
	ONH_SRL_MUSCLE_SIM_2015(QWidget *parent = 0);
	~ONH_SRL_MUSCLE_SIM_2015();

	void loadModels(void);

private:
	Ui::ONH_SRL_MUSCLE_SIM_2015Class ui;
	CustomGLVAOWidget* vaoWidget;
	QWidget *container;

protected:
	void keyPressEvent(QKeyEvent *event);
	void resizeEvent(QResizeEvent* event);
};

#endif // ONH_SRL_MUSCLE_SIM_2015_H
