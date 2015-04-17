/********************************************************************************
** Form generated from reading UI file 'onh_srl_muscle_sim_2015.ui'
**
** Created by: Qt User Interface Compiler version 5.4.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ONH_SRL_MUSCLE_SIM_2015_H
#define UI_ONH_SRL_MUSCLE_SIM_2015_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ONH_SRL_MUSCLE_SIM_2015Class
{
public:
    QWidget *centralWidget;
    QGroupBox *groupBox;
    QSlider *xRotSlider;
    QSlider *zRotSlider;
    QSlider *yRotSlider;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLCDNumber *lcdNumber;
    QLCDNumber *lcdNumber_2;
    QLCDNumber *lcdNumber_3;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ONH_SRL_MUSCLE_SIM_2015Class)
    {
        if (ONH_SRL_MUSCLE_SIM_2015Class->objectName().isEmpty())
            ONH_SRL_MUSCLE_SIM_2015Class->setObjectName(QStringLiteral("ONH_SRL_MUSCLE_SIM_2015Class"));
        ONH_SRL_MUSCLE_SIM_2015Class->resize(600, 320);
        centralWidget = new QWidget(ONH_SRL_MUSCLE_SIM_2015Class);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(20, 10, 231, 121));
        xRotSlider = new QSlider(groupBox);
        xRotSlider->setObjectName(QStringLiteral("xRotSlider"));
        xRotSlider->setGeometry(QRect(20, 30, 140, 20));
        xRotSlider->setMinimum(-180);
        xRotSlider->setMaximum(180);
        xRotSlider->setOrientation(Qt::Horizontal);
        xRotSlider->setTickPosition(QSlider::TicksAbove);
        xRotSlider->setTickInterval(20);
        zRotSlider = new QSlider(groupBox);
        zRotSlider->setObjectName(QStringLiteral("zRotSlider"));
        zRotSlider->setGeometry(QRect(20, 90, 140, 20));
        zRotSlider->setMinimum(-180);
        zRotSlider->setMaximum(180);
        zRotSlider->setTracking(true);
        zRotSlider->setOrientation(Qt::Horizontal);
        zRotSlider->setInvertedAppearance(false);
        zRotSlider->setInvertedControls(false);
        zRotSlider->setTickPosition(QSlider::TicksAbove);
        zRotSlider->setTickInterval(20);
        yRotSlider = new QSlider(groupBox);
        yRotSlider->setObjectName(QStringLiteral("yRotSlider"));
        yRotSlider->setGeometry(QRect(20, 60, 140, 20));
        yRotSlider->setMinimum(-180);
        yRotSlider->setMaximum(180);
        yRotSlider->setOrientation(Qt::Horizontal);
        yRotSlider->setTickPosition(QSlider::TicksAbove);
        yRotSlider->setTickInterval(20);
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(0, 30, 16, 16));
        label->setAlignment(Qt::AlignCenter);
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(0, 60, 16, 16));
        label_2->setAlignment(Qt::AlignCenter);
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(0, 90, 16, 16));
        label_3->setAlignment(Qt::AlignCenter);
        lcdNumber = new QLCDNumber(groupBox);
        lcdNumber->setObjectName(QStringLiteral("lcdNumber"));
        lcdNumber->setGeometry(QRect(170, 30, 51, 23));
        lcdNumber->setAutoFillBackground(false);
        lcdNumber->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber->setFrameShadow(QFrame::Raised);
        lcdNumber->setSmallDecimalPoint(false);
        lcdNumber->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_2 = new QLCDNumber(groupBox);
        lcdNumber_2->setObjectName(QStringLiteral("lcdNumber_2"));
        lcdNumber_2->setGeometry(QRect(170, 60, 51, 23));
        lcdNumber_2->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_2->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_3 = new QLCDNumber(groupBox);
        lcdNumber_3->setObjectName(QStringLiteral("lcdNumber_3"));
        lcdNumber_3->setGeometry(QRect(170, 90, 51, 23));
        lcdNumber_3->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_3->setSegmentStyle(QLCDNumber::Flat);
        ONH_SRL_MUSCLE_SIM_2015Class->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ONH_SRL_MUSCLE_SIM_2015Class);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        ONH_SRL_MUSCLE_SIM_2015Class->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ONH_SRL_MUSCLE_SIM_2015Class);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ONH_SRL_MUSCLE_SIM_2015Class->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ONH_SRL_MUSCLE_SIM_2015Class);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ONH_SRL_MUSCLE_SIM_2015Class->setStatusBar(statusBar);

        retranslateUi(ONH_SRL_MUSCLE_SIM_2015Class);
        QObject::connect(xRotSlider, SIGNAL(valueChanged(int)), lcdNumber, SLOT(display(int)));
        QObject::connect(yRotSlider, SIGNAL(valueChanged(int)), lcdNumber_2, SLOT(display(int)));
        QObject::connect(zRotSlider, SIGNAL(valueChanged(int)), lcdNumber_3, SLOT(display(int)));

        QMetaObject::connectSlotsByName(ONH_SRL_MUSCLE_SIM_2015Class);
    } // setupUi

    void retranslateUi(QMainWindow *ONH_SRL_MUSCLE_SIM_2015Class)
    {
        ONH_SRL_MUSCLE_SIM_2015Class->setWindowTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "ONH_SRL_MUSCLE_SIM_2015", 0));
        groupBox->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Rotation", 0));
        label->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_2->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_3->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
    } // retranslateUi

};

namespace Ui {
    class ONH_SRL_MUSCLE_SIM_2015Class: public Ui_ONH_SRL_MUSCLE_SIM_2015Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ONH_SRL_MUSCLE_SIM_2015_H
