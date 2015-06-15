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
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDial>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ONH_SRL_MUSCLE_SIM_2015Class
{
public:
    QAction *actionExit;
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
    QPushButton *resetView_button;
    QGroupBox *groupBox_4;
    QSlider *xCamPivotSlider;
    QSlider *zCamPivotSlider;
    QSlider *yCamPivotSlider;
    QLabel *label_11;
    QLabel *label_12;
    QLabel *label_13;
    QLCDNumber *xCamPivotLcdNumber;
    QLCDNumber *yCamPivotLcdNumber;
    QLCDNumber *zCamPivotLcdNumber;
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
    QGroupBox *groupBox_3;
    QSlider *PRONATOR_TERES_TranspSlider;
    QSlider *OTHER_TranspSlider;
    QCheckBox *BRACHIORADIALIS_checkBox;
    QSlider *TRICEPS_BRACHII_TranspSlider;
    QCheckBox *TRICEPS_BRACHII_checkBox;
    QCheckBox *ANCONEUS_checkBox;
    QSlider *BRACHIORADIALIS_TranspSlider;
    QCheckBox *BRACHIALIS_checkBox;
    QSlider *BRACHIALIS_TranspSlider;
    QCheckBox *BICEPS_BRACHII_checkBox;
    QLabel *label_9;
    QLabel *label_8;
    QSlider *ANCONEUS_TranspSlider;
    QCheckBox *PRONATOR_TERES_checkBox;
    QSlider *BICEPS_BRACHII_TranspSlider;
    QCheckBox *OTHER_checkBox;
    QCheckBox *ANCONEUS_wireframe_checkBox;
    QCheckBox *BRACHIALIS_wireframe_checkBox;
    QCheckBox *BRACHIORADIALIS_wireframe_checkBox;
    QCheckBox *PRONATOR_TERES_wireframe_checkBox;
    QCheckBox *BICEPS_BRACHII_wireframe_checkBox;
    QCheckBox *TRICEPS_BRACHII_wireframe_checkBox;
    QCheckBox *OTHER_wireframe_checkBox;
    QLabel *label_10;
    QCheckBox *BONES_checkBox;
    QSlider *BONES_TranspSlider;
    QCheckBox *BONES_wireframe_checkBox;
    QWidget *tab_3;
    QGroupBox *groupBox_5;
    QCheckBox *GRID_checkBox;
    QCheckBox *AXES_checkBox;
    QWidget *widget;
    QPlainTextEdit *logPlainTextEdit;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ONH_SRL_MUSCLE_SIM_2015Class)
    {
        if (ONH_SRL_MUSCLE_SIM_2015Class->objectName().isEmpty())
            ONH_SRL_MUSCLE_SIM_2015Class->setObjectName(QStringLiteral("ONH_SRL_MUSCLE_SIM_2015Class"));
        ONH_SRL_MUSCLE_SIM_2015Class->resize(1200, 700);
        actionExit = new QAction(ONH_SRL_MUSCLE_SIM_2015Class);
        actionExit->setObjectName(QStringLiteral("actionExit"));
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
        xCamPosSlider->setValue(30);
        xCamPosSlider->setOrientation(Qt::Horizontal);
        xCamPosSlider->setTickPosition(QSlider::TicksAbove);
        xCamPosSlider->setTickInterval(5);
        zCamPosSlider = new QSlider(groupBox_2);
        zCamPosSlider->setObjectName(QStringLiteral("zCamPosSlider"));
        zCamPosSlider->setGeometry(QRect(20, 90, 140, 20));
        zCamPosSlider->setMinimum(-50);
        zCamPosSlider->setMaximum(50);
        zCamPosSlider->setPageStep(5);
        zCamPosSlider->setValue(30);
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
        yCamPosSlider->setValue(15);
        yCamPosSlider->setOrientation(Qt::Horizontal);
        yCamPosSlider->setTickPosition(QSlider::TicksAbove);
        yCamPosSlider->setTickInterval(5);
        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(0, 30, 16, 16));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        label_4->setFont(font);
        label_4->setStyleSheet(QStringLiteral("color: rgb(255, 0, 0);"));
        label_4->setAlignment(Qt::AlignCenter);
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(0, 60, 16, 16));
        label_5->setFont(font);
        label_5->setStyleSheet(QStringLiteral("color: rgb(0, 255, 0);"));
        label_5->setAlignment(Qt::AlignCenter);
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(0, 90, 16, 16));
        label_6->setFont(font);
        label_6->setStyleSheet(QStringLiteral("color: rgb(0, 0, 255);"));
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
        lcdNumber_4->setProperty("value", QVariant(30));
        lcdNumber_5 = new QLCDNumber(groupBox_2);
        lcdNumber_5->setObjectName(QStringLiteral("lcdNumber_5"));
        lcdNumber_5->setGeometry(QRect(170, 60, 51, 23));
        lcdNumber_5->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_5->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_5->setProperty("value", QVariant(15));
        lcdNumber_6 = new QLCDNumber(groupBox_2);
        lcdNumber_6->setObjectName(QStringLiteral("lcdNumber_6"));
        lcdNumber_6->setGeometry(QRect(170, 90, 51, 23));
        lcdNumber_6->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_6->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_6->setProperty("value", QVariant(30));
        lcdNumber_6->setProperty("intValue", QVariant(30));
        fovyDial = new QDial(tab);
        fovyDial->setObjectName(QStringLiteral("fovyDial"));
        fovyDial->setGeometry(QRect(270, 30, 50, 64));
        fovyDial->setMinimum(-15);
        fovyDial->setMaximum(17);
        fovyDial->setPageStep(1);
        fovyDial->setValue(0);
        fovyDial->setNotchTarget(5);
        fovyDial->setNotchesVisible(true);
        label_7 = new QLabel(tab);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(270, 10, 51, 16));
        label_7->setAlignment(Qt::AlignCenter);
        lcdNumber_7 = new QLCDNumber(tab);
        lcdNumber_7->setObjectName(QStringLiteral("lcdNumber_7"));
        lcdNumber_7->setGeometry(QRect(270, 100, 51, 23));
        lcdNumber_7->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_7->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_7->setProperty("value", QVariant(0));
        resetView_button = new QPushButton(tab);
        resetView_button->setObjectName(QStringLiteral("resetView_button"));
        resetView_button->setGeometry(QRect(10, 270, 311, 23));
        groupBox_4 = new QGroupBox(tab);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        groupBox_4->setGeometry(QRect(10, 140, 231, 121));
        xCamPivotSlider = new QSlider(groupBox_4);
        xCamPivotSlider->setObjectName(QStringLiteral("xCamPivotSlider"));
        xCamPivotSlider->setGeometry(QRect(20, 30, 140, 20));
        xCamPivotSlider->setMinimum(-50);
        xCamPivotSlider->setMaximum(50);
        xCamPivotSlider->setPageStep(5);
        xCamPivotSlider->setValue(0);
        xCamPivotSlider->setOrientation(Qt::Horizontal);
        xCamPivotSlider->setTickPosition(QSlider::TicksAbove);
        xCamPivotSlider->setTickInterval(5);
        zCamPivotSlider = new QSlider(groupBox_4);
        zCamPivotSlider->setObjectName(QStringLiteral("zCamPivotSlider"));
        zCamPivotSlider->setGeometry(QRect(20, 90, 140, 20));
        zCamPivotSlider->setMinimum(-50);
        zCamPivotSlider->setMaximum(50);
        zCamPivotSlider->setPageStep(5);
        zCamPivotSlider->setValue(0);
        zCamPivotSlider->setTracking(true);
        zCamPivotSlider->setOrientation(Qt::Horizontal);
        zCamPivotSlider->setInvertedAppearance(false);
        zCamPivotSlider->setInvertedControls(false);
        zCamPivotSlider->setTickPosition(QSlider::TicksAbove);
        zCamPivotSlider->setTickInterval(5);
        yCamPivotSlider = new QSlider(groupBox_4);
        yCamPivotSlider->setObjectName(QStringLiteral("yCamPivotSlider"));
        yCamPivotSlider->setGeometry(QRect(20, 60, 140, 20));
        yCamPivotSlider->setMinimum(-50);
        yCamPivotSlider->setMaximum(50);
        yCamPivotSlider->setPageStep(5);
        yCamPivotSlider->setValue(0);
        yCamPivotSlider->setOrientation(Qt::Horizontal);
        yCamPivotSlider->setTickPosition(QSlider::TicksAbove);
        yCamPivotSlider->setTickInterval(5);
        label_11 = new QLabel(groupBox_4);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setGeometry(QRect(0, 30, 16, 16));
        label_11->setFont(font);
        label_11->setStyleSheet(QStringLiteral("color: rgb(255, 0, 0);"));
        label_11->setAlignment(Qt::AlignCenter);
        label_12 = new QLabel(groupBox_4);
        label_12->setObjectName(QStringLiteral("label_12"));
        label_12->setGeometry(QRect(0, 60, 16, 16));
        label_12->setFont(font);
        label_12->setStyleSheet(QStringLiteral("color: rgb(0, 255, 0);"));
        label_12->setAlignment(Qt::AlignCenter);
        label_13 = new QLabel(groupBox_4);
        label_13->setObjectName(QStringLiteral("label_13"));
        label_13->setGeometry(QRect(0, 90, 16, 16));
        label_13->setFont(font);
        label_13->setStyleSheet(QStringLiteral("color: rgb(0, 0, 255);"));
        label_13->setAlignment(Qt::AlignCenter);
        xCamPivotLcdNumber = new QLCDNumber(groupBox_4);
        xCamPivotLcdNumber->setObjectName(QStringLiteral("xCamPivotLcdNumber"));
        xCamPivotLcdNumber->setGeometry(QRect(170, 30, 51, 23));
        xCamPivotLcdNumber->setAutoFillBackground(false);
        xCamPivotLcdNumber->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        xCamPivotLcdNumber->setFrameShadow(QFrame::Raised);
        xCamPivotLcdNumber->setSmallDecimalPoint(false);
        xCamPivotLcdNumber->setSegmentStyle(QLCDNumber::Flat);
        xCamPivotLcdNumber->setProperty("value", QVariant(0));
        yCamPivotLcdNumber = new QLCDNumber(groupBox_4);
        yCamPivotLcdNumber->setObjectName(QStringLiteral("yCamPivotLcdNumber"));
        yCamPivotLcdNumber->setGeometry(QRect(170, 60, 51, 23));
        yCamPivotLcdNumber->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        yCamPivotLcdNumber->setSegmentStyle(QLCDNumber::Flat);
        yCamPivotLcdNumber->setProperty("value", QVariant(0));
        zCamPivotLcdNumber = new QLCDNumber(groupBox_4);
        zCamPivotLcdNumber->setObjectName(QStringLiteral("zCamPivotLcdNumber"));
        zCamPivotLcdNumber->setGeometry(QRect(170, 90, 51, 23));
        zCamPivotLcdNumber->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        zCamPivotLcdNumber->setSegmentStyle(QLCDNumber::Flat);
        zCamPivotLcdNumber->setProperty("value", QVariant(0));
        zCamPivotLcdNumber->setProperty("intValue", QVariant(0));
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        groupBox = new QGroupBox(tab_2);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 321, 121));
        xModelRotSlider = new QSlider(groupBox);
        xModelRotSlider->setObjectName(QStringLiteral("xModelRotSlider"));
        xModelRotSlider->setGeometry(QRect(40, 30, 191, 20));
        xModelRotSlider->setMinimum(-180);
        xModelRotSlider->setMaximum(180);
        xModelRotSlider->setValue(-160);
        xModelRotSlider->setOrientation(Qt::Horizontal);
        xModelRotSlider->setTickPosition(QSlider::TicksAbove);
        xModelRotSlider->setTickInterval(36);
        zModelRotSlider = new QSlider(groupBox);
        zModelRotSlider->setObjectName(QStringLiteral("zModelRotSlider"));
        zModelRotSlider->setGeometry(QRect(40, 90, 191, 20));
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
        yModelRotSlider->setGeometry(QRect(40, 60, 191, 20));
        yModelRotSlider->setMinimum(-180);
        yModelRotSlider->setMaximum(180);
        yModelRotSlider->setOrientation(Qt::Horizontal);
        yModelRotSlider->setTickPosition(QSlider::TicksAbove);
        yModelRotSlider->setTickInterval(36);
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(20, 30, 16, 16));
        label->setFont(font);
        label->setStyleSheet(QStringLiteral("color: rgb(255, 0, 0);"));
        label->setAlignment(Qt::AlignCenter);
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(20, 60, 16, 16));
        label_2->setFont(font);
        label_2->setStyleSheet(QStringLiteral("color: rgb(0, 255, 0);"));
        label_2->setAlignment(Qt::AlignCenter);
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(20, 90, 16, 16));
        label_3->setFont(font);
        label_3->setStyleSheet(QStringLiteral("color: rgb(0, 0, 255);"));
        label_3->setAlignment(Qt::AlignCenter);
        lcdNumber = new QLCDNumber(groupBox);
        lcdNumber->setObjectName(QStringLiteral("lcdNumber"));
        lcdNumber->setGeometry(QRect(250, 30, 51, 23));
        lcdNumber->setAutoFillBackground(false);
        lcdNumber->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber->setFrameShadow(QFrame::Raised);
        lcdNumber->setSmallDecimalPoint(false);
        lcdNumber->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber->setProperty("value", QVariant(-160));
        lcdNumber_2 = new QLCDNumber(groupBox);
        lcdNumber_2->setObjectName(QStringLiteral("lcdNumber_2"));
        lcdNumber_2->setGeometry(QRect(250, 60, 51, 23));
        lcdNumber_2->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_2->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber_3 = new QLCDNumber(groupBox);
        lcdNumber_3->setObjectName(QStringLiteral("lcdNumber_3"));
        lcdNumber_3->setGeometry(QRect(250, 90, 51, 23));
        lcdNumber_3->setStyleSheet(QLatin1String("color: rgb(0, 255, 0);\n"
"background-color: rgb(0, 0, 0);"));
        lcdNumber_3->setSegmentStyle(QLCDNumber::Flat);
        groupBox_3 = new QGroupBox(tab_2);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setGeometry(QRect(10, 140, 321, 231));
        PRONATOR_TERES_TranspSlider = new QSlider(groupBox_3);
        PRONATOR_TERES_TranspSlider->setObjectName(QStringLiteral("PRONATOR_TERES_TranspSlider"));
        PRONATOR_TERES_TranspSlider->setGeometry(QRect(160, 110, 100, 20));
        PRONATOR_TERES_TranspSlider->setMinimum(20);
        PRONATOR_TERES_TranspSlider->setMaximum(100);
        PRONATOR_TERES_TranspSlider->setSingleStep(1);
        PRONATOR_TERES_TranspSlider->setValue(100);
        PRONATOR_TERES_TranspSlider->setOrientation(Qt::Horizontal);
        PRONATOR_TERES_TranspSlider->setTickPosition(QSlider::NoTicks);
        PRONATOR_TERES_TranspSlider->setTickInterval(10);
        OTHER_TranspSlider = new QSlider(groupBox_3);
        OTHER_TranspSlider->setObjectName(QStringLiteral("OTHER_TranspSlider"));
        OTHER_TranspSlider->setGeometry(QRect(160, 170, 100, 20));
        OTHER_TranspSlider->setMinimum(20);
        OTHER_TranspSlider->setMaximum(100);
        OTHER_TranspSlider->setSingleStep(1);
        OTHER_TranspSlider->setValue(100);
        OTHER_TranspSlider->setOrientation(Qt::Horizontal);
        OTHER_TranspSlider->setTickPosition(QSlider::NoTicks);
        OTHER_TranspSlider->setTickInterval(10);
        BRACHIORADIALIS_checkBox = new QCheckBox(groupBox_3);
        BRACHIORADIALIS_checkBox->setObjectName(QStringLiteral("BRACHIORADIALIS_checkBox"));
        BRACHIORADIALIS_checkBox->setGeometry(QRect(20, 90, 130, 20));
        BRACHIORADIALIS_checkBox->setFont(font);
        BRACHIORADIALIS_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(0, 255, 0);"));
        BRACHIORADIALIS_checkBox->setChecked(true);
        TRICEPS_BRACHII_TranspSlider = new QSlider(groupBox_3);
        TRICEPS_BRACHII_TranspSlider->setObjectName(QStringLiteral("TRICEPS_BRACHII_TranspSlider"));
        TRICEPS_BRACHII_TranspSlider->setGeometry(QRect(160, 150, 100, 20));
        TRICEPS_BRACHII_TranspSlider->setMinimum(20);
        TRICEPS_BRACHII_TranspSlider->setMaximum(100);
        TRICEPS_BRACHII_TranspSlider->setSingleStep(1);
        TRICEPS_BRACHII_TranspSlider->setValue(100);
        TRICEPS_BRACHII_TranspSlider->setOrientation(Qt::Horizontal);
        TRICEPS_BRACHII_TranspSlider->setTickPosition(QSlider::NoTicks);
        TRICEPS_BRACHII_TranspSlider->setTickInterval(10);
        TRICEPS_BRACHII_checkBox = new QCheckBox(groupBox_3);
        TRICEPS_BRACHII_checkBox->setObjectName(QStringLiteral("TRICEPS_BRACHII_checkBox"));
        TRICEPS_BRACHII_checkBox->setGeometry(QRect(20, 150, 130, 20));
        TRICEPS_BRACHII_checkBox->setFont(font);
        TRICEPS_BRACHII_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(255, 255, 0);"));
        TRICEPS_BRACHII_checkBox->setChecked(true);
        ANCONEUS_checkBox = new QCheckBox(groupBox_3);
        ANCONEUS_checkBox->setObjectName(QStringLiteral("ANCONEUS_checkBox"));
        ANCONEUS_checkBox->setGeometry(QRect(20, 50, 130, 20));
        ANCONEUS_checkBox->setFont(font);
        ANCONEUS_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(255, 99, 45);"));
        ANCONEUS_checkBox->setChecked(true);
        BRACHIORADIALIS_TranspSlider = new QSlider(groupBox_3);
        BRACHIORADIALIS_TranspSlider->setObjectName(QStringLiteral("BRACHIORADIALIS_TranspSlider"));
        BRACHIORADIALIS_TranspSlider->setGeometry(QRect(160, 90, 100, 20));
        BRACHIORADIALIS_TranspSlider->setMinimum(20);
        BRACHIORADIALIS_TranspSlider->setMaximum(100);
        BRACHIORADIALIS_TranspSlider->setSingleStep(1);
        BRACHIORADIALIS_TranspSlider->setValue(100);
        BRACHIORADIALIS_TranspSlider->setOrientation(Qt::Horizontal);
        BRACHIORADIALIS_TranspSlider->setTickPosition(QSlider::NoTicks);
        BRACHIORADIALIS_TranspSlider->setTickInterval(10);
        BRACHIALIS_checkBox = new QCheckBox(groupBox_3);
        BRACHIALIS_checkBox->setObjectName(QStringLiteral("BRACHIALIS_checkBox"));
        BRACHIALIS_checkBox->setGeometry(QRect(20, 70, 130, 20));
        BRACHIALIS_checkBox->setFont(font);
        BRACHIALIS_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(255, 0, 255);"));
        BRACHIALIS_checkBox->setChecked(true);
        BRACHIALIS_TranspSlider = new QSlider(groupBox_3);
        BRACHIALIS_TranspSlider->setObjectName(QStringLiteral("BRACHIALIS_TranspSlider"));
        BRACHIALIS_TranspSlider->setGeometry(QRect(160, 70, 100, 20));
        BRACHIALIS_TranspSlider->setMinimum(20);
        BRACHIALIS_TranspSlider->setMaximum(100);
        BRACHIALIS_TranspSlider->setSingleStep(1);
        BRACHIALIS_TranspSlider->setValue(100);
        BRACHIALIS_TranspSlider->setOrientation(Qt::Horizontal);
        BRACHIALIS_TranspSlider->setTickPosition(QSlider::NoTicks);
        BRACHIALIS_TranspSlider->setTickInterval(10);
        BICEPS_BRACHII_checkBox = new QCheckBox(groupBox_3);
        BICEPS_BRACHII_checkBox->setObjectName(QStringLiteral("BICEPS_BRACHII_checkBox"));
        BICEPS_BRACHII_checkBox->setGeometry(QRect(20, 130, 130, 20));
        BICEPS_BRACHII_checkBox->setFont(font);
        BICEPS_BRACHII_checkBox->setStyleSheet(QLatin1String("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);"));
        BICEPS_BRACHII_checkBox->setChecked(true);
        label_9 = new QLabel(groupBox_3);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(20, 25, 130, 20));
        label_9->setAlignment(Qt::AlignCenter);
        label_8 = new QLabel(groupBox_3);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(160, 25, 100, 20));
        label_8->setAlignment(Qt::AlignCenter);
        ANCONEUS_TranspSlider = new QSlider(groupBox_3);
        ANCONEUS_TranspSlider->setObjectName(QStringLiteral("ANCONEUS_TranspSlider"));
        ANCONEUS_TranspSlider->setGeometry(QRect(160, 50, 100, 20));
        ANCONEUS_TranspSlider->setAutoFillBackground(false);
        ANCONEUS_TranspSlider->setStyleSheet(QStringLiteral("selection-color: rgb(255, 99, 45);"));
        ANCONEUS_TranspSlider->setMinimum(20);
        ANCONEUS_TranspSlider->setMaximum(100);
        ANCONEUS_TranspSlider->setSingleStep(1);
        ANCONEUS_TranspSlider->setValue(100);
        ANCONEUS_TranspSlider->setOrientation(Qt::Horizontal);
        ANCONEUS_TranspSlider->setInvertedControls(false);
        ANCONEUS_TranspSlider->setTickPosition(QSlider::NoTicks);
        ANCONEUS_TranspSlider->setTickInterval(10);
        PRONATOR_TERES_checkBox = new QCheckBox(groupBox_3);
        PRONATOR_TERES_checkBox->setObjectName(QStringLiteral("PRONATOR_TERES_checkBox"));
        PRONATOR_TERES_checkBox->setGeometry(QRect(20, 110, 130, 20));
        PRONATOR_TERES_checkBox->setFont(font);
        PRONATOR_TERES_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(0, 255, 255);"));
        PRONATOR_TERES_checkBox->setChecked(true);
        BICEPS_BRACHII_TranspSlider = new QSlider(groupBox_3);
        BICEPS_BRACHII_TranspSlider->setObjectName(QStringLiteral("BICEPS_BRACHII_TranspSlider"));
        BICEPS_BRACHII_TranspSlider->setGeometry(QRect(160, 130, 100, 20));
        BICEPS_BRACHII_TranspSlider->setMinimum(20);
        BICEPS_BRACHII_TranspSlider->setMaximum(100);
        BICEPS_BRACHII_TranspSlider->setSingleStep(1);
        BICEPS_BRACHII_TranspSlider->setValue(100);
        BICEPS_BRACHII_TranspSlider->setOrientation(Qt::Horizontal);
        BICEPS_BRACHII_TranspSlider->setTickPosition(QSlider::NoTicks);
        BICEPS_BRACHII_TranspSlider->setTickInterval(10);
        OTHER_checkBox = new QCheckBox(groupBox_3);
        OTHER_checkBox->setObjectName(QStringLiteral("OTHER_checkBox"));
        OTHER_checkBox->setGeometry(QRect(20, 170, 130, 20));
        OTHER_checkBox->setFont(font);
        OTHER_checkBox->setChecked(true);
        ANCONEUS_wireframe_checkBox = new QCheckBox(groupBox_3);
        ANCONEUS_wireframe_checkBox->setObjectName(QStringLiteral("ANCONEUS_wireframe_checkBox"));
        ANCONEUS_wireframe_checkBox->setGeometry(QRect(280, 50, 31, 17));
        ANCONEUS_wireframe_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(255, 99, 45);"));
        ANCONEUS_wireframe_checkBox->setChecked(false);
        BRACHIALIS_wireframe_checkBox = new QCheckBox(groupBox_3);
        BRACHIALIS_wireframe_checkBox->setObjectName(QStringLiteral("BRACHIALIS_wireframe_checkBox"));
        BRACHIALIS_wireframe_checkBox->setGeometry(QRect(280, 70, 31, 17));
        BRACHIALIS_wireframe_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(255, 0, 255);"));
        BRACHIALIS_wireframe_checkBox->setChecked(false);
        BRACHIORADIALIS_wireframe_checkBox = new QCheckBox(groupBox_3);
        BRACHIORADIALIS_wireframe_checkBox->setObjectName(QStringLiteral("BRACHIORADIALIS_wireframe_checkBox"));
        BRACHIORADIALIS_wireframe_checkBox->setGeometry(QRect(280, 90, 31, 17));
        BRACHIORADIALIS_wireframe_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(0, 255, 0);"));
        BRACHIORADIALIS_wireframe_checkBox->setChecked(false);
        PRONATOR_TERES_wireframe_checkBox = new QCheckBox(groupBox_3);
        PRONATOR_TERES_wireframe_checkBox->setObjectName(QStringLiteral("PRONATOR_TERES_wireframe_checkBox"));
        PRONATOR_TERES_wireframe_checkBox->setGeometry(QRect(280, 110, 31, 17));
        PRONATOR_TERES_wireframe_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(0, 255, 255);"));
        PRONATOR_TERES_wireframe_checkBox->setChecked(false);
        BICEPS_BRACHII_wireframe_checkBox = new QCheckBox(groupBox_3);
        BICEPS_BRACHII_wireframe_checkBox->setObjectName(QStringLiteral("BICEPS_BRACHII_wireframe_checkBox"));
        BICEPS_BRACHII_wireframe_checkBox->setGeometry(QRect(280, 130, 31, 17));
        BICEPS_BRACHII_wireframe_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(0, 0, 255);"));
        BICEPS_BRACHII_wireframe_checkBox->setChecked(false);
        TRICEPS_BRACHII_wireframe_checkBox = new QCheckBox(groupBox_3);
        TRICEPS_BRACHII_wireframe_checkBox->setObjectName(QStringLiteral("TRICEPS_BRACHII_wireframe_checkBox"));
        TRICEPS_BRACHII_wireframe_checkBox->setGeometry(QRect(280, 150, 31, 17));
        TRICEPS_BRACHII_wireframe_checkBox->setStyleSheet(QStringLiteral("background-color: rgb(255, 255, 0);"));
        TRICEPS_BRACHII_wireframe_checkBox->setChecked(false);
        OTHER_wireframe_checkBox = new QCheckBox(groupBox_3);
        OTHER_wireframe_checkBox->setObjectName(QStringLiteral("OTHER_wireframe_checkBox"));
        OTHER_wireframe_checkBox->setGeometry(QRect(280, 170, 31, 17));
        OTHER_wireframe_checkBox->setChecked(false);
        label_10 = new QLabel(groupBox_3);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(280, 25, 31, 20));
        label_10->setAlignment(Qt::AlignCenter);
        BONES_checkBox = new QCheckBox(groupBox_3);
        BONES_checkBox->setObjectName(QStringLiteral("BONES_checkBox"));
        BONES_checkBox->setGeometry(QRect(20, 190, 130, 20));
        BONES_checkBox->setFont(font);
        BONES_checkBox->setChecked(true);
        BONES_TranspSlider = new QSlider(groupBox_3);
        BONES_TranspSlider->setObjectName(QStringLiteral("BONES_TranspSlider"));
        BONES_TranspSlider->setGeometry(QRect(160, 190, 100, 20));
        BONES_TranspSlider->setMinimum(20);
        BONES_TranspSlider->setMaximum(100);
        BONES_TranspSlider->setSingleStep(1);
        BONES_TranspSlider->setValue(100);
        BONES_TranspSlider->setOrientation(Qt::Horizontal);
        BONES_TranspSlider->setTickPosition(QSlider::NoTicks);
        BONES_TranspSlider->setTickInterval(10);
        BONES_wireframe_checkBox = new QCheckBox(groupBox_3);
        BONES_wireframe_checkBox->setObjectName(QStringLiteral("BONES_wireframe_checkBox"));
        BONES_wireframe_checkBox->setGeometry(QRect(280, 190, 31, 17));
        BONES_wireframe_checkBox->setChecked(false);
        tabWidget->addTab(tab_2, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        groupBox_5 = new QGroupBox(tab_3);
        groupBox_5->setObjectName(QStringLiteral("groupBox_5"));
        groupBox_5->setGeometry(QRect(10, 10, 321, 121));
        GRID_checkBox = new QCheckBox(groupBox_5);
        GRID_checkBox->setObjectName(QStringLiteral("GRID_checkBox"));
        GRID_checkBox->setGeometry(QRect(10, 30, 70, 17));
        GRID_checkBox->setChecked(true);
        AXES_checkBox = new QCheckBox(groupBox_5);
        AXES_checkBox->setObjectName(QStringLiteral("AXES_checkBox"));
        AXES_checkBox->setGeometry(QRect(10, 60, 70, 17));
        AXES_checkBox->setChecked(true);
        tabWidget->addTab(tab_3, QString());
        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(370, 30, 800, 600));
        widget->setStyleSheet(QStringLiteral("background-color: rgb(0, 0, 0);"));
        logPlainTextEdit = new QPlainTextEdit(centralWidget);
        logPlainTextEdit->setObjectName(QStringLiteral("logPlainTextEdit"));
        logPlainTextEdit->setGeometry(QRect(10, 480, 350, 150));
        QFont font1;
        font1.setFamily(QStringLiteral("Courier New"));
        font1.setPointSize(8);
        logPlainTextEdit->setFont(font1);
        logPlainTextEdit->viewport()->setProperty("cursor", QVariant(QCursor(Qt::IBeamCursor)));
        logPlainTextEdit->setStyleSheet(QLatin1String("background-color: rgb(0, 0, 0);\n"
"color: rgb(0, 255, 0);"));
        logPlainTextEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        logPlainTextEdit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        logPlainTextEdit->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        logPlainTextEdit->setLineWrapMode(QPlainTextEdit::NoWrap);
        logPlainTextEdit->setOverwriteMode(true);
        logPlainTextEdit->setBackgroundVisible(false);
        logPlainTextEdit->setCenterOnScroll(true);
        ONH_SRL_MUSCLE_SIM_2015Class->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ONH_SRL_MUSCLE_SIM_2015Class);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1200, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        ONH_SRL_MUSCLE_SIM_2015Class->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ONH_SRL_MUSCLE_SIM_2015Class);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ONH_SRL_MUSCLE_SIM_2015Class->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ONH_SRL_MUSCLE_SIM_2015Class);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ONH_SRL_MUSCLE_SIM_2015Class->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionExit);

        retranslateUi(ONH_SRL_MUSCLE_SIM_2015Class);
        QObject::connect(fovyDial, SIGNAL(valueChanged(int)), lcdNumber_7, SLOT(display(int)));
        QObject::connect(xCamPosSlider, SIGNAL(valueChanged(int)), lcdNumber_4, SLOT(display(int)));
        QObject::connect(xModelRotSlider, SIGNAL(valueChanged(int)), lcdNumber, SLOT(display(int)));
        QObject::connect(yCamPosSlider, SIGNAL(valueChanged(int)), lcdNumber_5, SLOT(display(int)));
        QObject::connect(yModelRotSlider, SIGNAL(valueChanged(int)), lcdNumber_2, SLOT(display(int)));
        QObject::connect(zCamPosSlider, SIGNAL(valueChanged(int)), lcdNumber_6, SLOT(display(int)));
        QObject::connect(zModelRotSlider, SIGNAL(valueChanged(int)), lcdNumber_3, SLOT(display(int)));
        QObject::connect(xCamPivotSlider, SIGNAL(valueChanged(int)), xCamPivotLcdNumber, SLOT(display(int)));
        QObject::connect(yCamPivotSlider, SIGNAL(valueChanged(int)), yCamPivotLcdNumber, SLOT(display(int)));
        QObject::connect(zCamPivotSlider, SIGNAL(valueChanged(int)), zCamPivotLcdNumber, SLOT(display(int)));

        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(ONH_SRL_MUSCLE_SIM_2015Class);
    } // setupUi

    void retranslateUi(QMainWindow *ONH_SRL_MUSCLE_SIM_2015Class)
    {
        ONH_SRL_MUSCLE_SIM_2015Class->setWindowTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "ONH_SRL_MUSCLE_SIM_2015", 0));
        actionExit->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Exit", 0));
        actionExit->setShortcut(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Alt+E", 0));
        groupBox_2->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Position", 0));
        label_4->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_5->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_6->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
        label_7->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Zoom", 0));
        resetView_button->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Reset View", 0));
        groupBox_4->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Pivot", 0));
        label_11->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_12->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_13->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Camera", 0));
        groupBox->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Global Rotation (degrees)", 0));
        label->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_2->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_3->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
        groupBox_3->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Parts", 0));
        BRACHIORADIALIS_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "BRACHIORADIALIS", 0));
        TRICEPS_BRACHII_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "TRICEPS BRACHII", 0));
        ANCONEUS_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "ANCONEUS", 0));
        BRACHIALIS_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "BRACHIALIS", 0));
        BICEPS_BRACHII_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "BICEPS BRACHII", 0));
        label_9->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Display", 0));
        label_8->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Opacity (%)", 0));
        PRONATOR_TERES_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "PRONATOR TERES", 0));
        OTHER_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "OTHER MUSCLES", 0));
        ANCONEUS_wireframe_checkBox->setText(QString());
        BRACHIALIS_wireframe_checkBox->setText(QString());
        BRACHIORADIALIS_wireframe_checkBox->setText(QString());
        PRONATOR_TERES_wireframe_checkBox->setText(QString());
        BICEPS_BRACHII_wireframe_checkBox->setText(QString());
        TRICEPS_BRACHII_wireframe_checkBox->setText(QString());
        OTHER_wireframe_checkBox->setText(QString());
        label_10->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Wire", 0));
        BONES_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "BONES", 0));
        BONES_wireframe_checkBox->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Model", 0));
        groupBox_5->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Global Helpers", 0));
        GRID_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Grid", 0));
        AXES_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Axes", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Helpers", 0));
        logPlainTextEdit->setDocumentTitle(QString());
        logPlainTextEdit->setPlainText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Hello", 0));
        menuFile->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "File", 0));
    } // retranslateUi

};

namespace Ui {
    class ONH_SRL_MUSCLE_SIM_2015Class: public Ui_ONH_SRL_MUSCLE_SIM_2015Class {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ONH_SRL_MUSCLE_SIM_2015_H
