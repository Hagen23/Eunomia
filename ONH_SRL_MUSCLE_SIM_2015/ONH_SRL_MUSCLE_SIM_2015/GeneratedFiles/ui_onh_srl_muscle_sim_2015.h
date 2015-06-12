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
    QSlider *OTHER_TranspSlider;
    QLabel *label_8;
    QCheckBox *ANCONEUS_checkBox;
    QCheckBox *BRACHIALIS_checkBox;
    QCheckBox *BRACHIORDIALIS_checkBox;
    QCheckBox *PRONATOR_TERES_checkBox;
    QCheckBox *BICEPS_BRACHII_checkBox;
    QCheckBox *TRICEPS_BRACHII_checkBox;
    QSlider *ANCONEUS_TranspSlider;
    QCheckBox *OTHER_checkBox;
    QSlider *BRACHIALIS_TranspSlider;
    QSlider *BRACHIORDIALIS_TranspSlider;
    QSlider *PRONATOR_TERES_TranspSlider;
    QSlider *BICEPS_BRACHII_TranspSlider;
    QSlider *TRICEPS_BRACHII_TranspSlider;
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
        fovyDial->setMinimum(-15);
        fovyDial->setMaximum(17);
        fovyDial->setPageStep(1);
        fovyDial->setValue(0);
        fovyDial->setNotchTarget(2);
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
        resetView_button->setGeometry(QRect(10, 140, 311, 23));
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
        OTHER_TranspSlider = new QSlider(tab_2);
        OTHER_TranspSlider->setObjectName(QStringLiteral("OTHER_TranspSlider"));
        OTHER_TranspSlider->setGeometry(QRect(160, 330, 140, 20));
        OTHER_TranspSlider->setMinimum(0);
        OTHER_TranspSlider->setMaximum(100);
        OTHER_TranspSlider->setSingleStep(1);
        OTHER_TranspSlider->setValue(50);
        OTHER_TranspSlider->setOrientation(Qt::Horizontal);
        OTHER_TranspSlider->setTickPosition(QSlider::TicksAbove);
        OTHER_TranspSlider->setTickInterval(10);
        label_8 = new QLabel(tab_2);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(180, 180, 101, 20));
        ANCONEUS_checkBox = new QCheckBox(tab_2);
        ANCONEUS_checkBox->setObjectName(QStringLiteral("ANCONEUS_checkBox"));
        ANCONEUS_checkBox->setGeometry(QRect(30, 210, 120, 20));
        ANCONEUS_checkBox->setChecked(true);
        BRACHIALIS_checkBox = new QCheckBox(tab_2);
        BRACHIALIS_checkBox->setObjectName(QStringLiteral("BRACHIALIS_checkBox"));
        BRACHIALIS_checkBox->setGeometry(QRect(30, 230, 120, 20));
        BRACHIALIS_checkBox->setChecked(true);
        BRACHIORDIALIS_checkBox = new QCheckBox(tab_2);
        BRACHIORDIALIS_checkBox->setObjectName(QStringLiteral("BRACHIORDIALIS_checkBox"));
        BRACHIORDIALIS_checkBox->setGeometry(QRect(30, 250, 120, 20));
        BRACHIORDIALIS_checkBox->setChecked(true);
        PRONATOR_TERES_checkBox = new QCheckBox(tab_2);
        PRONATOR_TERES_checkBox->setObjectName(QStringLiteral("PRONATOR_TERES_checkBox"));
        PRONATOR_TERES_checkBox->setGeometry(QRect(30, 270, 120, 20));
        PRONATOR_TERES_checkBox->setChecked(true);
        BICEPS_BRACHII_checkBox = new QCheckBox(tab_2);
        BICEPS_BRACHII_checkBox->setObjectName(QStringLiteral("BICEPS_BRACHII_checkBox"));
        BICEPS_BRACHII_checkBox->setGeometry(QRect(30, 290, 120, 20));
        BICEPS_BRACHII_checkBox->setChecked(true);
        TRICEPS_BRACHII_checkBox = new QCheckBox(tab_2);
        TRICEPS_BRACHII_checkBox->setObjectName(QStringLiteral("TRICEPS_BRACHII_checkBox"));
        TRICEPS_BRACHII_checkBox->setGeometry(QRect(30, 310, 120, 20));
        TRICEPS_BRACHII_checkBox->setChecked(true);
        ANCONEUS_TranspSlider = new QSlider(tab_2);
        ANCONEUS_TranspSlider->setObjectName(QStringLiteral("ANCONEUS_TranspSlider"));
        ANCONEUS_TranspSlider->setGeometry(QRect(160, 210, 140, 20));
        ANCONEUS_TranspSlider->setMinimum(0);
        ANCONEUS_TranspSlider->setMaximum(100);
        ANCONEUS_TranspSlider->setSingleStep(1);
        ANCONEUS_TranspSlider->setValue(50);
        ANCONEUS_TranspSlider->setOrientation(Qt::Horizontal);
        ANCONEUS_TranspSlider->setTickPosition(QSlider::TicksAbove);
        ANCONEUS_TranspSlider->setTickInterval(10);
        OTHER_checkBox = new QCheckBox(tab_2);
        OTHER_checkBox->setObjectName(QStringLiteral("OTHER_checkBox"));
        OTHER_checkBox->setGeometry(QRect(30, 330, 120, 20));
        OTHER_checkBox->setChecked(true);
        BRACHIALIS_TranspSlider = new QSlider(tab_2);
        BRACHIALIS_TranspSlider->setObjectName(QStringLiteral("BRACHIALIS_TranspSlider"));
        BRACHIALIS_TranspSlider->setGeometry(QRect(160, 230, 140, 20));
        BRACHIALIS_TranspSlider->setMinimum(0);
        BRACHIALIS_TranspSlider->setMaximum(100);
        BRACHIALIS_TranspSlider->setSingleStep(1);
        BRACHIALIS_TranspSlider->setValue(50);
        BRACHIALIS_TranspSlider->setOrientation(Qt::Horizontal);
        BRACHIALIS_TranspSlider->setTickPosition(QSlider::TicksAbove);
        BRACHIALIS_TranspSlider->setTickInterval(10);
        BRACHIORDIALIS_TranspSlider = new QSlider(tab_2);
        BRACHIORDIALIS_TranspSlider->setObjectName(QStringLiteral("BRACHIORDIALIS_TranspSlider"));
        BRACHIORDIALIS_TranspSlider->setGeometry(QRect(160, 250, 140, 20));
        BRACHIORDIALIS_TranspSlider->setMinimum(0);
        BRACHIORDIALIS_TranspSlider->setMaximum(100);
        BRACHIORDIALIS_TranspSlider->setSingleStep(1);
        BRACHIORDIALIS_TranspSlider->setValue(50);
        BRACHIORDIALIS_TranspSlider->setOrientation(Qt::Horizontal);
        BRACHIORDIALIS_TranspSlider->setTickPosition(QSlider::TicksAbove);
        BRACHIORDIALIS_TranspSlider->setTickInterval(10);
        PRONATOR_TERES_TranspSlider = new QSlider(tab_2);
        PRONATOR_TERES_TranspSlider->setObjectName(QStringLiteral("PRONATOR_TERES_TranspSlider"));
        PRONATOR_TERES_TranspSlider->setGeometry(QRect(160, 270, 140, 20));
        PRONATOR_TERES_TranspSlider->setMinimum(0);
        PRONATOR_TERES_TranspSlider->setMaximum(100);
        PRONATOR_TERES_TranspSlider->setSingleStep(1);
        PRONATOR_TERES_TranspSlider->setValue(50);
        PRONATOR_TERES_TranspSlider->setOrientation(Qt::Horizontal);
        PRONATOR_TERES_TranspSlider->setTickPosition(QSlider::TicksAbove);
        PRONATOR_TERES_TranspSlider->setTickInterval(10);
        BICEPS_BRACHII_TranspSlider = new QSlider(tab_2);
        BICEPS_BRACHII_TranspSlider->setObjectName(QStringLiteral("BICEPS_BRACHII_TranspSlider"));
        BICEPS_BRACHII_TranspSlider->setGeometry(QRect(160, 290, 140, 20));
        BICEPS_BRACHII_TranspSlider->setMinimum(0);
        BICEPS_BRACHII_TranspSlider->setMaximum(100);
        BICEPS_BRACHII_TranspSlider->setSingleStep(1);
        BICEPS_BRACHII_TranspSlider->setValue(50);
        BICEPS_BRACHII_TranspSlider->setOrientation(Qt::Horizontal);
        BICEPS_BRACHII_TranspSlider->setTickPosition(QSlider::TicksAbove);
        BICEPS_BRACHII_TranspSlider->setTickInterval(10);
        TRICEPS_BRACHII_TranspSlider = new QSlider(tab_2);
        TRICEPS_BRACHII_TranspSlider->setObjectName(QStringLiteral("TRICEPS_BRACHII_TranspSlider"));
        TRICEPS_BRACHII_TranspSlider->setGeometry(QRect(160, 310, 140, 20));
        TRICEPS_BRACHII_TranspSlider->setMinimum(0);
        TRICEPS_BRACHII_TranspSlider->setMaximum(100);
        TRICEPS_BRACHII_TranspSlider->setSingleStep(1);
        TRICEPS_BRACHII_TranspSlider->setValue(50);
        TRICEPS_BRACHII_TranspSlider->setOrientation(Qt::Horizontal);
        TRICEPS_BRACHII_TranspSlider->setTickPosition(QSlider::TicksAbove);
        TRICEPS_BRACHII_TranspSlider->setTickInterval(10);
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

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(ONH_SRL_MUSCLE_SIM_2015Class);
    } // setupUi

    void retranslateUi(QMainWindow *ONH_SRL_MUSCLE_SIM_2015Class)
    {
        ONH_SRL_MUSCLE_SIM_2015Class->setWindowTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "ONH_SRL_MUSCLE_SIM_2015", 0));
        groupBox_2->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Position", 0));
        label_4->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_5->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_6->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
        label_7->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "ZOOM", 0));
        resetView_button->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Reset View", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Camera", 0));
        groupBox->setTitle(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Rotation", 0));
        label->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "X", 0));
        label_2->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Y", 0));
        label_3->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Z", 0));
        label_8->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "Transparency (%):", 0));
        ANCONEUS_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "ANCONEUS", 0));
        BRACHIALIS_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "BRACHIALIS", 0));
        BRACHIORDIALIS_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "BRACHIORDIALIS", 0));
        PRONATOR_TERES_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "PRONATOR TERES", 0));
        BICEPS_BRACHII_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "BICEPS BRACHII", 0));
        TRICEPS_BRACHII_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "TRICEPS BRACHII", 0));
        OTHER_checkBox->setText(QApplication::translate("ONH_SRL_MUSCLE_SIM_2015Class", "OTHER GEOMETRY", 0));
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
