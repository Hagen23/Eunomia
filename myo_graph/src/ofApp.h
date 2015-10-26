#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include <vector>
#include <fstream>
//For saving the timestamp of the data
#include <ctime> 
#include <iostream>
//Second way of getting timestamp
#include <chrono>

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		void recordData();

		ofxPanel gui;
		

		ofxLabel label_channel1_1;
		ofxLabel label_channel1_2;
		ofxLabel label_channel1_3;
		ofxLabel label_channel1_4;
		ofxLabel label_channel1_5;
		ofxLabel label_channel1_6;
		ofxLabel label_channel1_7;
		ofxLabel label_channel1_8;

		ofxLabel label_channel2_1;
		ofxLabel label_channel2_2;
		ofxLabel label_channel2_3;
		ofxLabel label_channel2_4;
		ofxLabel label_channel2_5;
		ofxLabel label_channel2_6;
		ofxLabel label_channel2_7;
		ofxLabel label_channel2_8;

		std::vector<double> points_channel_1_1;
		std::vector<double> points_channel_1_2;
		std::vector<double> points_channel_1_3;
		std::vector<double> points_channel_1_4;
		std::vector<double> points_channel_1_5;
		std::vector<double> points_channel_1_6;
		std::vector<double> points_channel_1_7;
		std::vector<double> points_channel_1_8;

		std::vector<double> points_channel_2_1;
		std::vector<double> points_channel_2_2;
		std::vector<double> points_channel_2_3;
		std::vector<double> points_channel_2_4;
		std::vector<double> points_channel_2_5;
		std::vector<double> points_channel_2_6;
		std::vector<double> points_channel_2_7;
		std::vector<double> points_channel_2_8;
		
		
		typedef struct DATAREC
		{
			string timestamp;
			int time;
			char channel1_1;
			char channel1_2;
			char channel1_3;
			char channel1_4;
			char channel1_5;
			char channel1_6;
			char channel1_7;
			char channel1_8;

			char channel2_1;
			char channel2_2;
			char channel2_3;
			char channel2_4;
			char channel2_5;
			char channel2_6;
			char channel2_7;
			char channel2_8;
		}sensorRegistry;

		std::vector<sensorRegistry> dataSet;

		bool record;
		bool recording;
};
