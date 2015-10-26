#pragma once

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <time.h>
#include <myo\myo.hpp>
#include <myo\cxx\Quaternion.hpp>
#include <myo\cxx\Vector3.hpp>
#include <myo\cxx\Pose.hpp>
#include <stdio.h>
#include <math.h>


class MyoDataCollector : public myo::DeviceListener
{
public:
	MyoDataCollector() : emgSamples1(), emgSamples2()
	{

		calibrated_1 = false;
		calibrated_2 = false;
		opened = false;
		_antiYaw = myo::Quaternion<float>();
		_referenceRoll = 0.0f;
		openFiles();
		_lastPose = myo::Pose::Type::unknown;
		vibrate_count = 0;
	}
	

	void MyoDataCollector::openFiles();
	
	std::vector<myo::Myo*> knownMyos;
	
	myo::Quaternion<float> _antiYaw;
	float _referenceRoll;
	myo::Pose _lastPose;
	int vibrate_count;

	
	float roll_1;
	float yaw_1;
	float pitch_1;


	float roll_2;
	float yaw_2;
	float pitch_2;


	//myo::Vector3<float> computeZeroRollVector(myo::Vector3<float> forward);
	// onUnpair() is called whenever the Myo is disconnected from Myo Connect by the user.
	void onUnpair(myo::Myo* myo, uint64_t timestamp);
	float MyoDataCollector::rollFromZero(myo::Vector3<float> zeroRoll, myo::Vector3<float> forward, myo::Vector3<float> up);
	
	// onEmgData() is called whenever a paired Myo has provided new EMG data, and EMG streaming is enabled.
	void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg);
	
	void onPair(myo::Myo* myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion);

	void registerNewMyo(myo::Myo* myo);

	//void registerNewMyo(myo::Myo* myo);
	// There are other virtual functions in DeviceListener that we could override here, like onAccelerometerData().
	// For this example, the functions overridden above are sufficient.

	// We define this function to print the current values that were updated by the on...() functions above.
	void print();
	int getMyoNumber(myo::Myo* myo);
	size_t identifyMyo(myo::Myo* myo);
	void onOrientationData(myo::Myo *myo, uint64_t timestamp, const myo::Quaternion< float > &rotation);
	void onAccelerometerData(myo::Myo *myo, uint64_t timestamp, const myo::Vector3< float > &accel);
	void onGyroscopeData(myo::Myo *myo, uint64_t timestamp, const myo::Vector3< float > &gyro);
	void onConnect(myo::Myo *myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion);
	void printVector(std::ofstream &file, uint64_t timestamp, const myo::Vector3< float > &vector);

	// The values of this array is set by onEmgData() above.
	std::array<int8_t, 8> emgSamples1;
	std::array<int8_t, 8> emgSamples2;
	// The files we are logging to
	std::ofstream emgFile1;
	std::ofstream gyroFile1;
	std::ofstream orientationFile1;
	std::ofstream orientationEulerFile1;
	std::ofstream accelerometerFile1;
	std::ofstream emgFile2;
	std::ofstream gyroFile2;
	std::ofstream orientationFile2;
	std::ofstream orientationEulerFile2;
	std::ofstream accelerometerFile2;

	bool opened;
	bool calibrated_1;
	bool calibrated_2;
};


