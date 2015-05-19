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
#include <QtWidgets/QDial>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ONH_SRL_MUSCLE_SIM_2015Class
{
public:
    QWidget *centralWidget;
    QTabWidget *tabWidget;
    QWidget *tab;
    QGroupBox *groupBox_2;
    QSlider *xCamPosSlider;
    QSlider *zCamPosSlider;
    QSlider *yCamPosSlider;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *label_6;
    QLCDNumber *lcdNumber_4;
    QLCDNumber *lcdNumber_5;
    QLCDNumber *lcdNumber_6;
    QDial *fovyDial;
    QLabel *label_7;
    QLCDNumber *lcdNumber_7;
    QWidget *tab_2;
    QGroupBox *groupBox;
    QSlider *xModelRotSlider;
    QSlider *zModelRotSlider;
    QSlider *yModelRotSlider;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLCDNumber *lcdNumber;
    QLCDNumber *lcdNumber_2;
    QLCDNumber *lcdNumber_3;
    QSlider *modelTranspSlider;
    QLabel *label_8;
    QLCDNumber *lcdNumber_8;
    QWidget *widget;
    QPlainTextEdit *logPlainTextEdit;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ONH_SRL_MUSCLE_SIM_2015Class)
    {
        if (ONH_SRL_MUSCLE_SIM_2015Class->objectName().isEmpty())
            ONH_SRL_MUSCLE_SIM_2015Class->setObjectName(QStringLiteral("ONH_SRL_MUSCLE_SIM_2015Class"));
        ONH_SRL_MUSCLE_SIM_2015Class->resize(1200, 700);
        centralWidget = new QWidget(ONH_SRL_MUSCLE_SIM_2015Class);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(10, 10, 350, 460));
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        groupBox_2 = new QGroupBox(tab);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 10, 231, 121));
        xCamPosSlider = new QSlider(groupBox_2);
        xCamPosSlider->setObjectName(QStringLiteral("xCamPosSlider"));
        xCamPosSlider->setGeometry(QRect(20, 30, 140, 20));
        xCamPosSlider->setMinimum(-50);
        xCamPosSlider->setMaximum(50);
        xCamPosSlider->setPageStep(5);
        xCamPosSlider->setValue(10);
        xCamPosSlider->setOrientation(Qt::Horizontal);
        xCamPosSlider->setTickPosition(QSlider::TicksAbove);
        xCamPosSlider->setTickInterval(5);
        zCamPosSlider = new QSlider(groupBox_2);
        zCamPosSlider->setObjectName(QStringLiteral("zCamPosSlider"));
        zCamPosSlider->setGeometry(QRect(20, 90, 140, 20));
        zCamPosSlider->setMinimum(-50);
        zCamPosSlider->setMaximum(50);
        zCamPosSlider->setPageStep(5);
        zCamPosSlider->setValue(40);
        zCamPosSlider->setTracking(true);
        zCamPosSlider->setOrientation(Qt::Horizontal);
        zCamPosSlider->setInvertedAppearance(false);
        zCamPosSlider->setInvertedControls(false);
        zCamPosSlider->setTickPosition(QSlider::TicksAbove);
        zCamPosSlider->setTickInterval(5);
        yCamPosSlider = new QSlider(groupBox_2);
        yCamPosSlider->setObjectName(QStringLiteral("yCamPosSlider"));
        yCamPosSlider->setGeometry(QRect(20, 60, 140, 20));
        yCamPosSlider->setMinimum(-50);
        yCamPosSlider->setMaximum(50);
        yCamPosSlider->setPageStep(5);
        yCamPosSlider->setValue(10);
        yCamPosSlider->setOrientation(Qt::Horizontal);
        yCamPosSlider->setTickPosition(QSlider::TicksAbove);
        yCamPosSlider->setTickInterval(5);
        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(0, 30, 16, 16));
        label_4->setAlignment(Qt::AlignCenter);
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(0, 60, 16, 16));
        label_5->setAlignment(Qt::AlignCenter);
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(0, 90, 16, 16));
        label_6->setAlignment(Qt::AlignCenter);
        lcdNumber_4 = new QLCDNumber(groupBox_2);
        lcdNumber_4->setObjectName(QStringLiteral("lcdNumber_4"));
        lcdNumber_4->setGeometry(QRect(170, 30, 51, 23));
        lcdNumber_4->setAutoFillBackground(false);
        lcdNumber_4->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_4->setFrameShadow(QFrame::Raised);
        lcdNumber_4->setSmallDecimalPoint(false);
        lcdNumber_4->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_4->setProperty("value", QVariant(10));
        lcdNumber_5 = new QLCDNumber(groupBox_2);
        lcdNumber_5->setObjectName(QStringLiteral("lcdNumber_5"));
        lcdNumber_5->setGeometry(QRect(170, 60, 51, 23));
        lcdNumber_5->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_5->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_5->setProperty("value", QVariant(10));
        lcdNumber_6 = new QLCDNumber(groupBox_2);
        lcdNumber_6->setObjectName(QStringLiteral("lcdNumber_6"));
        lcdNumber_6->setGeometry(QRect(170, 90, 51, 23));
        lcdNumber_6->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_6->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_6->setProperty("value", QVariant(40));
        fovyDial = new QDial(tab);
        fovyDial->setObjectName(QStringLiteral("fovyDial"));
        fovyDial->setGeometry(QRect(270, 30, 50, 64));
        fovyDial->setMinimum(30);
        fovyDial->setMaximum(100);
        fovyDial->setPageStep(5);
        fovyDial->setValue(44);
        label_7 = new QLabel(tab);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(270, 20, 51, 16));
        label_7->setAlignment(Qt::AlignCenter);
        lcdNumber_7 = new QLCDNumber(tab);
        lcdNumber_7->setObjectName(QStringLiteral("lcdNumber_7"));
        lcdNumber_7->setGeometry(QRect(270, 100, 51, 23));
        lcdNumber_7->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_7->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_7->setProperty("value", QVariant(44));
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        groupBox = new QGroupBox(tab_2);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 231, 121));
        xModelRotSlider = new QSlider(groupBox);
        xModelRotSlider->setObjectName(QStringLiteral("xModelRotSlider"));
        xModelRotSlider->setGeometry(QRect(20, 30, 140, 20));
        xModelRotSlider->setMinimum(-180);
        xModelRotSlider->setMaximum(180);
        xModelRotSlider->setValue(-160);
        xModelRotSlider->setOrientation(Qt::Horizontal);
        xModelRotSlider->setTickPosition(QSlider::TicksAbove);
        xModelRotSlider->setTickInterval(36);
        zModelRotSlider = new QSlider(groupBox);
        zModelRotSlider->setObjectName(QStringLiteral("zModelRotSlider"));
        zModelRotSlider->setGeometry(QRect(20, 90, 140, 20));
        zModelRotSlider->setMinimum(-180);
        zModelRotSlider->setMaximum(180);
        zModelRotSlider->setTracking(true);
        zModelRotSlider->setOrientation(Qt::Horizontal);
        zModelRotSlider->setInvertedAppearance(false);
        zModelRotSlider->setInvertedControls(false);
        zModelRotSlider->setTickPosition(QSlider::TicksAbove);
        zModelRotSlider->setTickInterval(36);
        yModelRotSlider = new QSlider(groupBox);
        yModelRotSlider->setObjectName(QStringLiteral("yModelRotSlider"));
        yModelRotSlider->setGeometry(QRect(20, 60, 140, 20));
        yModelRotSlider->setMinimum(-180);
        yModelRotSlider->setMaximum(180);
        yModelRotSlider->setOrientation(Qt::Horizontal);
        yModelRotSlider->setTickPosition(QSlider::TicksAbove);
        yModelRotSlider->setTickInterval(36);
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
        lcdNumber->setProperty("value", QVariant(-160));
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
        modelTranspSlider = new QSlider(tab_2);
        modelTranspSlider->setObjectName(QStringLiteral("modelTranspSlider"));
        modelTranspSlider->setGeometry(QRect(110, 150, 140, 20));
        modelTranspSlider->setMinimum(0);
        modelTranspSlider->setMaximum(200);
        modelTranspSlider->setSingleStep(1);
        modelTranspSlider->setValue(50);
        modelTranspSlider->setOrientation(Qt::Horizontal);
        modelTranspSlider->setTickPosition(QSlider::TicksAbove);
        modelTranspSlider->setTickInterval(10);
        label_8 = new QLabel(tab_2);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(10, 150, 101, 20));
        lcdNumber_8 = new QLCDNumber(tab_2);
        lcdNumber_8->setObjectName(QStringLiteral("lcdNumber_8"));
        lcdNumber_8->setGeometry(QRect(260, 150, 51, 23));
        lcdNumber_8->setAutoFillBackground(false);
        lcdNumber_8->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_8->setFrameShadow(QFrame::Raised);
        lcdNumber_8->setSmallDecimalPoint(false);
        lcdNumber_8->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_8->setProperty("value", QVariant(50));
        tabWidget->addTab(tab_2, QString());
        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(370, 30, 800, 600));
        widget->setStyleSheet(QStringLiteral("background-color: rgb(0, 0, 0);"));
        logPlainTextEdit = new QPlainTextEdit(centralWidget);
        logPlainTextEdit->setObjectName(QStringLiteral("logPlainTextEdit"));
        logPlainTextEdit->setGeometry(QRect(10, 480, 350, 150));
        QFont font;
        font.setFamily(QStringLiteral("Courier New"));
        font.setPointSize(8);
        logPlainTextEdit->setFont(font);
        logPlainTextEdit->setStyleSheet(QLatin1String("background-color: rgb(0, 0, 0);\n"
"color: rgb(0, 255, 0);"));
        logPlainTextEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        logPlainTextEdit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        logPlainTextEdit->setLineWrapMode(QPlainTextEdit::NoWrap);
        ONH_SRL_MUSCLE_SIM_2015Class->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ONH_SRL_MUSCLE_SIM_2015Class);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1200, 21));
        ONH_SRL_MUSCLE_SIM_2015Class->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ONH_SRL_MUSCLE_SIM_2015Class);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ONH_SRL_MUSCLE_SIM_2015Class->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ONH_SRL_MUSCLE_SIM_2015Class);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ONH_SRL_MUSCLE_SIM_2015Class->setStatusBar(statusBar);

        retranslateUi(ONH_SRL_MUSCLE_SIM_2015Class);
        QObject::connect(fovyDial, SIGNAL(valueChanged(int)), lcdNumber_7, SLOT(display(int)));
        QObject::connect(xCamPosSlider, SIGNAL(valueChanged(int)), lcdNumber_4, SLOT(display(int)));
        QObject::connect(xModelRotSlider, SIGNAL(valueChanged(int)), lcdNumber, SLOT(display(int)));
        QObject::connect(yCamPosSlider, SIGNAL(valueChanged(int)), lcdNumber_5, SLOT(display(int)));
        QObject::connect(yModelRotSlider, SIGNAL(valueChanged(int)), lcdNumber_2, SLOT(display(int)));
        QObject::connect(zCamPosSlider, SIGNAL(valueChanged(int)), lcdNumber_6, SLOT(display(int)));
        QObject::connect(zModelRotSlider, SIGNAL(valueChanged(int)), lcdNumber_3, SLOT(display(int)));
        QObject::connect(modelTranspSlider, SIGNAL(valueChanged(int)), lcdNumber_8, SLOT(display(int)));

        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(ONH_SRL_MUSCLE_SIM_2015Class);
    } // setupUi

    void retranslateUi(QMainWindow *ONH_SRL_MUSCLE_SIM_2015Class)
    {
        ONH_SRL_MUSCLE_SIM_2015Class->setWindowTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "ONH_SRL_MUSCLE_SIM_2015", 0));
        groupBox_2->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Position", 0));
        label_4->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_5->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_6->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
        label_7->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "FOVY", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Camera", 0));
        groupBox->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Rotation", 0));
        label->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_2->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_3->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
        label_8->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Transparency (%):", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Model", 0));
        logPlainTextEdit->setDocumentTitle(QString());
        logPlainTextEdit->setPlainText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Hello", 0));
    } // retranslateUi

};

namespace Ui {
    class ONH_SRL_MUSCLE_SIM_2015Class: public Ui_ONH_SRL_MUSCLE_SIM_2015Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ONH_SRL_MUSCLE_SIM_2015_H
