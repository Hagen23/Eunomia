#include <iostream>
#include <stdexcept>
#include <vector>
#include "ofApp.h"
#include "MyoDataCollector.h"
#include <myo/myo.hpp>


//Temp multiple myos
/*class PrintMyoEvents : public myo::DeviceListener {
public:
	// Every time Myo Connect successfully pairs with a Myo armband, this function will be called.
	//
	// You can rely on the following rules:
	//  - onPair() will only be called once for each Myo device
	//  - no other events will occur involving a given Myo device before onPair() is called with it
	//
	// If you need to do some kind of per-Myo preparation before handling events, you can safely do it in onPair().
	void onPair(myo::Myo* myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion)
	{
		// Print out the MAC address of the armband we paired with.

		// The pointer address we get for a Myo is unique - in other words, it's safe to compare two Myo pointers to
		// see if they're referring to the same Myo.

		// Add the Myo pointer to our list of known Myo devices. This list is used to implement identifyMyo() below so
		// that we can give each Myo a nice short identifier.
		knownMyos.push_back(myo);

		// Now that we've added it to our list, get our short ID for it and print it out.
		std::cout << "Paired with " << identifyMyo(myo) << "." << std::endl;
	}

	void onPose(myo::Myo* myo, uint64_t timestamp, myo::Pose pose)
	{
		std::cout << "Myo " << identifyMyo(myo) << " switched to pose " << pose.toString() << "." << std::endl;
	}

	void onConnect(myo::Myo* myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion)
	{
		std::cout << "Myo " << identifyMyo(myo) << " has connected." << std::endl;
	}

	void onDisconnect(myo::Myo* myo, uint64_t timestamp)
	{
		std::cout << "Myo " << identifyMyo(myo) << " has disconnected." << std::endl;
	}

	// This is a utility function implemented for this sample that maps a myo::Myo* to a unique ID starting at 1.
	// It does so by looking for the Myo pointer in knownMyos, which onPair() adds each Myo into as it is paired.
	size_t identifyMyo(myo::Myo* myo) {
		// Walk through the list of Myo devices that we've seen pairing events for.
		for (size_t i = 0; i < knownMyos.size(); ++i) {
			// If two Myo pointers compare equal, they refer to the same Myo device.
			if (knownMyos[i] == myo) {
				return i + 1;
			}
		}

		return 0;
	}

	// We store each Myo pointer that we pair with in this list, so that we can keep track of the order we've seen
	// each Myo and give it a unique short identifier (see onPair() and identifyMyo() above).
	std::vector<myo::Myo*> knownMyos;
};*/
//End Tem multiple myos

//com.example.emg-data-sample
myo::Hub hub("com.example.multiple-myos");
myo::Hub hub2("com.example.multiple-myos");
MyoDataCollector collector;
//MyoDataCollector collector2;
/*
std::vector<myo::Myo*> knownMyos;//temp


// This is a utility function implemented for this sample that maps a myo::Myo* to a unique ID starting at 1.
// It does so by looking for the Myo pointer in knownMyos, which onPair() adds each Myo into as it is paired.
size_t identifyMyo(myo::Myo* myo) {
	// Walk through the list of Myo devices that we've seen pairing events for.
	for (size_t i = 0; i < knownMyos.size(); ++i) {
		// If two Myo pointers compare equal, they refer to the same Myo device.
		if (knownMyos[i] == myo) {
			return i + 1;
		}
	}

	return 0;
}

void onPair(myo::Myo* myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion)
{
	// Print out the MAC address of the armband we paired with.

	// The pointer address we get for a Myo is unique - in other words, it's safe to compare two Myo pointers to
	// see if they're referring to the same Myo.

	// Add the Myo pointer to our list of known Myo devices. This list is used to implement identifyMyo() below so
	// that we can give each Myo a nice short identifier.
	knownMyos.push_back(myo);

	// Now that we've added it to our list, get our short ID for it and print it out.
	std::cout << "Paired with " << identifyMyo(myo) << "." << std::endl;
}
*/
//--------------------------------------------------------------
void ofApp::setup()
{
	// We catch any exceptions that might occur below -- see the catch statement for more details.
	try {
		// Instantiate the PrintMyoEvents class we defined above, and attach it as a listener to our Hub.
		/*PrintMyoEvents printer;
		hub.addListener(&printer);

		hub.run(10);
		hub.run(10);
		hub.run(10);
		hub.run(10);
		*/
		/*
		//Temp multiple
		myo::Hub hub("com.example.multiple-myos");

		// Instantiate the PrintMyoEvents class we defined above, and attach it as a listener to our Hub.
		PrintMyoEvents printer;
		hub.addListener(&printer);

		while (1) {
			// Process events for 10 milliseconds at a time.
			cout << hub.run(10).;
		}
		//End temp multiple
		*/
		// First, we create a Hub with our application identifier. Be sure not to use the com.example namespace when
		// publishing your application. The Hub provides access to one or more Myos.
		

		std::cout << "Attempting to find a Myo..." << std::endl;

		// Next, we attempt to find a Myo to use. If a Myo is already paired in Myo Connect, this will return that Myo
		// immediately.
		// waitForMyo() takes a timeout value in milliseconds. In this case we will try to find a Myo for 10 seconds, and
		// if that fails, the function will return a null pointer.

		myo::Myo* myo;
		myo::Myo* myo2;

		//while (!myo)
		//{
		//	cout << "Estoy buscando";
			myo = hub.waitForMyo(10000);
		//}
		//while (!myo)
		//{
		//	cout << "Estoy buscando";
			myo2 = hub.waitForMyo(10000);
		//}
		cout << myo << "\n";
		cout << myo2 << "\n";
		
		if (myo)
			collector.registerNewMyo(myo);
		if (myo2)
			collector.registerNewMyo(myo2);
		
		// If waitForMyo() returned a null pointer, we failed to find a Myo, so exit with an error message.
		if (!myo) {
			throw std::runtime_error("Unable to find first Myo!");
		}
		else
		{
			cout << "Se conectó ";// << identifyMyo(myo);
		}
		if (!myo2) {
			throw std::runtime_error("Unable to find second Myo!");
		}

		// We've found a Myo.
		std::cout << "Connected to a Myo armband!" << std::endl << std::endl;

		// Next we enable EMG streaming on the found Myo.
		if (myo)
			myo->setStreamEmg(myo::Myo::streamEmgEnabled);

		if (myo2)
			myo2->setStreamEmg(myo::Myo::streamEmgEnabled);

		// Next we construct an instance of our DeviceListener, so that we can register it with the Hub.
		
		// Hub::addListener() takes the address of any object whose class inherits from DeviceListener, and will cause
		// Hub::run() to send events to all registered device listeners.
		hub.addListener(&collector);
		//hub.addListener(&collector2);
		
		label_channel1_1.setup("Channel", "1.1", 100, 50);
		label_channel1_1.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_1.setPosition(ofPoint(ofVec2f(200, 20)));

		label_channel1_2.setup("Channel", "1.2", 100, 50);
		label_channel1_2.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_2.setPosition(ofPoint(ofVec2f(200,120)));

		label_channel1_3.setup("Channel", "1.3", 100, 50);
		label_channel1_3.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_3.setPosition(ofPoint(ofVec2f(200, 220)));

		label_channel1_4.setup("Channel", "1.4", 100, 50);
		label_channel1_4.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_4.setPosition(ofPoint(ofVec2f(200, 320)));

		label_channel1_5.setup("Channel", "1.5", 100, 50);
		label_channel1_5.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_5.setPosition(ofPoint(ofVec2f(600, 20)));

		label_channel1_6.setup("Channel", "1.6", 100, 50);
		label_channel1_6.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_6.setPosition(ofPoint(ofVec2f(600, 120)));

		label_channel1_7.setup("Channel", "1.7", 100, 50);
		label_channel1_7.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_7.setPosition(ofPoint(ofVec2f(600, 220)));

		label_channel1_8.setup("Channel", "1.8", 100, 50);
		label_channel1_8.setFillColor(ofColor(0, 0, 0, 255));

		label_channel1_8.setPosition(ofPoint(ofVec2f(600, 320)));

		
		label_channel2_1.setup("Channel", "2.1", 100, 50);
		label_channel2_1.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_1.setPosition(ofPoint(ofVec2f(200, 420)));

		label_channel2_2.setup("Channel", "2.2", 100, 50);
		label_channel2_2.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_2.setPosition(ofPoint(ofVec2f(200, 520)));

		label_channel2_3.setup("Channel", "2.3", 100, 50);
		label_channel2_3.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_3.setPosition(ofPoint(ofVec2f(200, 620)));

		label_channel2_4.setup("Channel", "2.4", 100, 50);
		label_channel2_5.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_4.setPosition(ofPoint(ofVec2f(200, 720)));

		label_channel2_5.setup("Channel", "2.5", 100, 50);
		label_channel2_5.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_5.setPosition(ofPoint(ofVec2f(600, 420)));

		label_channel2_6.setup("Channel", "2.6", 100, 50);
		label_channel2_6.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_6.setPosition(ofPoint(ofVec2f(600, 520)));

		label_channel2_7.setup("Channel", "2.7", 100, 50);
		label_channel2_7.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_7.setPosition(ofPoint(ofVec2f(600, 620)));

		label_channel2_8.setup("Channel", "2.8", 100, 50);
		label_channel2_8.setFillColor(ofColor(0, 0, 0, 255));

		label_channel2_8.setPosition(ofPoint(ofVec2f(600, 720)));
		
		record = false;
		recording = false;
		
		// If a standard exception occurred, we print out its message and exit.
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		std::cerr << "Press enter to continue.";
		std::cin.ignore();
		exit();
	}

	
	/*label_channel1.setTextColor(ofColor(255, 255, 255));
	label_channel2.setTextColor(ofColor(255, 255, 255));
	label_channel3.setTextColor(ofColor(255, 255, 255));
	label_channel4.setTextColor(ofColor(255, 255, 255));
	label_channel5.setTextColor(ofColor(255, 255, 255));
	label_channel6.setTextColor(ofColor(255, 255, 255));
	label_channel7.setTextColor(ofColor(255, 255, 255));
	label_channel8.setTextColor(ofColor(255, 255, 255));*/

	//

}

//--------------------------------------------------------------
void ofApp::update()
{
	static int i = 0;
	static int time = 0;
	hub.run(1000 / 20);
	//hub2.run(1000 / 20);
	double m = -0.546875;
	double b = 80.0;

	//We substract values just to adjust the vertical position of the values
	points_channel_1_1.push_back((m*(double)collector.emgSamples1[0] + b) - 40);
	points_channel_1_2.push_back((m*(double)collector.emgSamples1[1] + b) + 60);
	points_channel_1_3.push_back((m*(double)collector.emgSamples1[2] + b) + 160);
	points_channel_1_4.push_back((m*(double)collector.emgSamples1[3] + b) + 260);
	points_channel_1_5.push_back((m*(double)collector.emgSamples1[4] + b) - 40);
	points_channel_1_6.push_back((m*(double)collector.emgSamples1[5] + b) + 60);
	points_channel_1_7.push_back((m*(double)collector.emgSamples1[6] + b) + 160);
	points_channel_1_8.push_back((m*(double)collector.emgSamples1[7] + b) + 260);

	
	points_channel_2_1.push_back((m*(double)collector.emgSamples2[0] + b) + 360);
	points_channel_2_2.push_back((m*(double)collector.emgSamples2[1] + b) + 460);
	points_channel_2_3.push_back((m*(double)collector.emgSamples2[2] + b) + 560);
	points_channel_2_4.push_back((m*(double)collector.emgSamples2[3] + b) + 660);
	points_channel_2_5.push_back((m*(double)collector.emgSamples2[4] + b) + 360);
	points_channel_2_6.push_back((m*(double)collector.emgSamples2[5] + b) + 460);
	points_channel_2_7.push_back((m*(double)collector.emgSamples2[6] + b) + 560);
	points_channel_2_8.push_back((m*(double)collector.emgSamples2[7] + b) + 660);
	
	++i;
	
	if (i >= 200)
	{
		i--;
		points_channel_1_1.erase(points_channel_1_1.begin());
		points_channel_1_2.erase(points_channel_1_2.begin());
		points_channel_1_3.erase(points_channel_1_3.begin());
		points_channel_1_4.erase(points_channel_1_4.begin());
		points_channel_1_5.erase(points_channel_1_5.begin());
		points_channel_1_6.erase(points_channel_1_6.begin());
		points_channel_1_7.erase(points_channel_1_7.begin());
		points_channel_1_8.erase(points_channel_1_8.begin());


		points_channel_2_1.erase(points_channel_2_1.begin());
		points_channel_2_2.erase(points_channel_2_2.begin());
		points_channel_2_3.erase(points_channel_2_3.begin());
		points_channel_2_4.erase(points_channel_2_4.begin());
		points_channel_2_5.erase(points_channel_2_5.begin());
		points_channel_2_6.erase(points_channel_2_6.begin());
		points_channel_2_7.erase(points_channel_2_7.begin());
		points_channel_2_8.erase(points_channel_2_8.begin());
	}

	if (record)
	{

		char buffer[30];
		std::stringstream  currentTime;
		SYSTEMTIME st;
		GetLocalTime(&st);
		currentTime << setfill('0') << std::setw(2) << st.wHour << ':'
			<< std::setw(2) << st.wMinute << ':'
			<< std::setw(2) << st.wSecond << '.'
			<< std::setw(3) << st.wMilliseconds;
		

		sensorRegistry reg = { 
			currentTime.str(),
			time,
			collector.emgSamples1[0],
			collector.emgSamples1[1],
			collector.emgSamples1[2],
			collector.emgSamples1[3],
			collector.emgSamples1[4],
			collector.emgSamples1[5],
			collector.emgSamples1[6],
			collector.emgSamples1[7],
			collector.emgSamples2[0],
			collector.emgSamples2[1],
			collector.emgSamples2[2],
			collector.emgSamples2[3],
			collector.emgSamples2[4],
			collector.emgSamples2[5],
			collector.emgSamples2[6],
			collector.emgSamples2[7]
		};

		dataSet.push_back(reg);
		time++;
		recording = true;
	}
	else if (!record && recording)
	{
		recordData();
		time = 0;
		recording = false;
	}
	else
	{
		time = 0;
	}

}

void ofApp::recordData()
{
	std::ofstream stream;
	stream.open("sensor.csv");
	
	for (int i = 0; i < dataSet.size(); ++i)
	{
		sensorRegistry reg=dataSet.at(i);
		stream << reg.timestamp << "," << reg.time << "," 
			<< (int)reg.channel1_1 << ","
			<< (int)reg.channel1_2 << ","
			<< (int)reg.channel1_3 << ","
			<< (int)reg.channel1_4 << ","
			<< (int)reg.channel1_5 << ","
			<< (int)reg.channel1_6 << ","
			<< (int)reg.channel1_7 << ","
			<< (int)reg.channel1_8 << ","
			<< (int)reg.channel2_1 << ","
			<< (int)reg.channel2_2 << ","
			<< (int)reg.channel2_3 << ","
			<< (int)reg.channel2_4 << ","
			<< (int)reg.channel2_5 << ","
			<< (int)reg.channel2_6 << ","
			<< (int)reg.channel2_7 << ","
			<< (int)reg.channel2_8 << ","
			<< std::endl;
	}
	stream.close();
	dataSet.clear();

}

//--------------------------------------------------------------
void ofApp::draw()
{
	ofPolyline poly_channel_1_1;
	ofPolyline poly_channel_1_2;
	ofPolyline poly_channel_1_3;
	ofPolyline poly_channel_1_4;
	ofPolyline poly_channel_1_5;
	ofPolyline poly_channel_1_6;
	ofPolyline poly_channel_1_7;
	ofPolyline poly_channel_1_8;

	ofPolyline poly_channel_2_1;
	ofPolyline poly_channel_2_2;
	ofPolyline poly_channel_2_3;
	ofPolyline poly_channel_2_4;
	ofPolyline poly_channel_2_5;
	ofPolyline poly_channel_2_6;
	ofPolyline poly_channel_2_7;
	ofPolyline poly_channel_2_8;

	collector.print();
	ofBackground(ofColor(0, 0, 0));
	ofSetLineWidth(2.0);

	ofSetColor(ofColor(0, 255, 0));

	for (int i = 0; i < points_channel_1_1.size(); ++i)
	{
		poly_channel_1_1.addVertex(ofPoint(i, points_channel_1_1.at(i)));
	}

	poly_channel_1_1.draw();
	poly_channel_1_1.clear();

	///
	ofSetColor(ofColor(255, 0, 0));

	for (int i = 0; i < points_channel_1_2.size(); ++i)
	{
		poly_channel_1_2.addVertex(ofPoint(i, points_channel_1_2.at(i)));
	}

	poly_channel_1_2.draw();
	poly_channel_1_2.clear();

	///
	ofSetColor(ofColor(0, 0, 255));
	
	for (int i = 0; i < points_channel_1_3.size(); ++i)
	{
		poly_channel_1_3.addVertex(ofPoint(i, points_channel_1_3.at(i)));
	}

	poly_channel_1_3.draw();
	poly_channel_1_3.clear();

	//
	ofSetColor(ofColor(255, 0, 255));

	for (int i = 0; i < points_channel_1_4.size(); ++i)
	{
		poly_channel_1_4.addVertex(ofPoint(i, points_channel_1_4.at(i)));
	}

	poly_channel_1_4.draw();
	poly_channel_1_4.clear();

	//
	ofSetColor(ofColor(0, 255, 255));

	for (int i = 0; i < points_channel_1_5.size(); ++i)
	{
		poly_channel_1_5.addVertex(ofPoint(i + 400, points_channel_1_5.at(i)));
	}

	poly_channel_1_5.draw();
	poly_channel_1_5.clear();

	//
	ofSetColor(ofColor(255, 255, 0));

	for (int i = 0; i < points_channel_1_6.size(); ++i)
	{
		poly_channel_1_6.addVertex(ofPoint(i + 400, points_channel_1_6.at(i)));
	}

	poly_channel_1_6.draw();
	poly_channel_1_6.clear();

	//
	ofSetColor(ofColor(255, 255, 255));

	for (int i = 0; i < points_channel_1_7.size(); ++i)
	{
		poly_channel_1_7.addVertex(ofPoint(i + 400, points_channel_1_7.at(i)));
	}

	poly_channel_1_7.draw();
	poly_channel_1_7.clear();

	//
	ofSetColor(ofColor(128,128, 128));

	for (int i = 0; i < points_channel_1_8.size(); ++i)
	{
		poly_channel_1_8.addVertex(ofPoint(i + 400, points_channel_1_8.at(i)));
	}

	poly_channel_1_8.draw();
	poly_channel_1_8.clear();

	for (int i = 0; i < points_channel_2_1.size(); ++i)
	{
		poly_channel_2_1.addVertex(ofPoint(i, points_channel_2_1.at(i)));
	}

	poly_channel_2_1.draw();
	poly_channel_2_1.clear();

	///
	ofSetColor(ofColor(255, 0, 0));

	for (int i = 0; i < points_channel_2_2.size(); ++i)
	{
		poly_channel_2_2.addVertex(ofPoint(i, points_channel_2_2.at(i)));
	}

	poly_channel_2_2.draw();
	poly_channel_2_2.clear();

	///
	ofSetColor(ofColor(0, 0, 255));

	for (int i = 0; i < points_channel_2_3.size(); ++i)
	{
		poly_channel_2_3.addVertex(ofPoint(i, points_channel_2_3.at(i)));
	}

	poly_channel_2_3.draw();
	poly_channel_2_3.clear();

	//
	ofSetColor(ofColor(255, 0, 255));

	for (int i = 0; i < points_channel_2_4.size(); ++i)
	{
		poly_channel_2_4.addVertex(ofPoint(i, points_channel_2_4.at(i)));
	}

	poly_channel_2_4.draw();
	poly_channel_2_4.clear();

	//
	ofSetColor(ofColor(0, 255, 255));

	for (int i = 0; i < points_channel_2_5.size(); ++i)
	{
		poly_channel_2_5.addVertex(ofPoint(i + 400, points_channel_2_5.at(i)));
	}

	poly_channel_2_5.draw();
	poly_channel_2_5.clear();

	//
	ofSetColor(ofColor(255, 255, 0));

	for (int i = 0; i < points_channel_2_6.size(); ++i)
	{
		poly_channel_2_6.addVertex(ofPoint(i + 400, points_channel_2_6.at(i)));
	}

	poly_channel_2_6.draw();
	poly_channel_2_6.clear();

	//
	ofSetColor(ofColor(255, 255, 255));

	for (int i = 0; i < points_channel_2_7.size(); ++i)
	{
		poly_channel_2_7.addVertex(ofPoint(i + 400, points_channel_2_7.at(i)));
	}

	poly_channel_2_7.draw();
	poly_channel_2_7.clear();

	//
	ofSetColor(ofColor(128, 128, 128));

	for (int i = 0; i < points_channel_2_8.size(); ++i)
	{
		poly_channel_2_8.addVertex(ofPoint(i + 400, points_channel_2_8.at(i)));
	}

	poly_channel_2_8.draw();
	poly_channel_2_8.clear();

	label_channel1_1.draw();
	label_channel1_2.draw();
	label_channel1_3.draw();
	label_channel1_4.draw();
	label_channel1_5.draw();
	label_channel1_6.draw();
	label_channel1_7.draw();
	label_channel1_8.draw();

	label_channel2_1.draw();
	label_channel2_2.draw();
	label_channel2_3.draw();
	label_channel2_4.draw();
	label_channel2_5.draw();
	label_channel2_6.draw();
	label_channel2_7.draw();
	label_channel2_8.draw();
	
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
	if (key == 'r')
	{
		record = record == true ? false : true;
		if (record)
		{
			std::cout << "Recording" << std::endl;
		}
		else
		{
			std::cout << "Stop Recording" << std::endl;
		}
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key)
{

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y )
{

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button)
{

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button)
{

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button)
{

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h)
{

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg)
{

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo)
{ 

}
