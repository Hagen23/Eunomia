#include "onh_srl_muscle_sim_2015.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	ONH_SRL_MUSCLE_SIM_2015 w;
	w.show();
	return a.exec();
}
