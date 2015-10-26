#include "MyoDataCollector.h"


// This is a utility function implemented for this sample that maps a myo::Myo* to a unique ID starting at 1.
// It does so by looking for the Myo pointer in knownMyos, which onPair() adds each Myo into as it is paired.
size_t MyoDataCollector::identifyMyo(myo::Myo* myo) {
	// Walk through the list of Myo devices that we've seen pairing events for.
	for (size_t i = 0; i < MyoDataCollector::knownMyos.size(); ++i) {
		// If two Myo pointers compare equal, they refer to the same Myo device.
		if (MyoDataCollector::knownMyos[i] == myo) {
			return i + 1;
		}
	}

	return 0;
}



void MyoDataCollector::openFiles() {
	time_t timestamp = std::time(0);

	//First myo
	// Open file for EMG log
	if (emgFile1.is_open()) {
		emgFile1.close();
	}
	std::ostringstream emgFileString1;
	emgFileString1 << "emg1-" << timestamp << ".csv";
	emgFile1.open(emgFileString1.str(), std::ios::out);
	emgFile1 << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8" << std::endl;

	// Open file for gyroscope log
	if (gyroFile1.is_open()) {
		gyroFile1.close();
	}
	std::ostringstream gyroFileString1;
	gyroFileString1 << "gyro1-" << timestamp << ".csv";
	gyroFile1.open(gyroFileString1.str(), std::ios::out);
	gyroFile1 << "timestamp,x,y,z" << std::endl;

	// Open file for accelerometer log
	if (accelerometerFile1.is_open()) {
		accelerometerFile1.close();
	}
	std::ostringstream accelerometerFileString1;
	accelerometerFileString1 << "accelerometer1-" << timestamp << ".csv";
	accelerometerFile1.open(accelerometerFileString1.str(), std::ios::out);
	accelerometerFile1 << "timestamp,x,y,z" << std::endl;

	// Open file for orientation log
	if (orientationFile1.is_open()) {
		orientationFile1.close();
	}
	std::ostringstream orientationFileString1;
	orientationFileString1 << "orientation1-" << timestamp << ".csv";
	orientationFile1.open(orientationFileString1.str(), std::ios::out);
	orientationFile1 << "timestamp,x,y,z,w" << std::endl;

	// Open file for orientation (Euler angles) log
	if (orientationEulerFile1.is_open()) {
		orientationEulerFile1.close();
	}
	std::ostringstream orientationEulerFileString1;
	orientationEulerFileString1 << "orientationEuler1-" << timestamp << ".csv";
	orientationEulerFile1.open(orientationEulerFileString1.str(), std::ios::out);
	orientationEulerFile1 << "timestamp,roll,pitch,yaw" << std::endl;


	//Second myo
	// Open file for EMG log
	if (emgFile2.is_open()) {
		emgFile2.close();
	}
	std::ostringstream emgFileString2;
	emgFileString2 << "emg2-" << timestamp << ".csv";
	emgFile2.open(emgFileString2.str(), std::ios::out);
	emgFile2 << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8" << std::endl;

	// Open file for gyroscope log
	if (gyroFile2.is_open()) {
		gyroFile2.close();
	}
	std::ostringstream gyroFileString2;
	gyroFileString2 << "gyro2-" << timestamp << ".csv";
	gyroFile2.open(gyroFileString2.str(), std::ios::out);
	gyroFile2 << "timestamp,x,y,z" << std::endl;

	// Open file for accelerometer log
	if (accelerometerFile2.is_open()) {
		accelerometerFile2.close();
	}
	std::ostringstream accelerometerFileString2;
	accelerometerFileString2 << "accelerometer2-" << timestamp << ".csv";
	accelerometerFile2.open(accelerometerFileString2.str(), std::ios::out);
	accelerometerFile2 << "timestamp,x,y,z" << std::endl;

	// Open file for orientation log
	if (orientationFile2.is_open()) {
		orientationFile2.close();
	}
	std::ostringstream orientationFileString2;
	orientationFileString2 << "orientation2-" << timestamp << ".csv";
	orientationFile2.open(orientationFileString2.str(), std::ios::out);
	orientationFile2 << "timestamp,x,y,z,w" << std::endl;

	// Open file for orientation (Euler angles) log
	if (orientationEulerFile2.is_open()) {
		orientationEulerFile2.close();
	}
	std::ostringstream orientationEulerFileString2;
	orientationEulerFileString2 << "orientationEuler2-" << timestamp << ".csv";
	orientationEulerFile2.open(orientationEulerFileString2.str(), std::ios::out);
	orientationEulerFile2 << "timestamp,roll,pitch,yaw" << std::endl;

}


//Start of new methods

// onEmgData() is called whenever a paired Myo has provided new EMG data, and EMG streaming is enabled.
void MyoDataCollector::onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg)
{

	
	int myoNumber = getMyoNumber(myo);
	if (myoNumber == 0)
	{
		emgFile1 << timestamp;
		for (size_t i = 0; i < 8; i++) {
			emgFile1 << ',' << static_cast<int>(emg[i]);

		}
		emgFile1 << std::endl;
		for (int i = 0; i < 8; i++) {
			emgSamples1[i] = emg[i];
		}
	}
	else if (myoNumber == 1)
	{
		emgFile2 << timestamp;
		for (size_t i = 0; i < 8; i++) {
			emgFile2 << ',' << static_cast<int>(emg[i]);

		}
		emgFile2 << std::endl;
		for (int i = 0; i < 8; i++) {
			emgSamples2[i] = emg[i];
		}

		vibrate_count++;
		if (vibrate_count > 1000)
		{
			myo->vibrate(myo::Myo::VibrationType::vibrationShort);
			vibrate_count = 0;
		}
	}
}

// onOrientationData is called whenever new orientation data is provided
// Be warned: This will not make any distiction between data from other Myo armbands
void MyoDataCollector::onOrientationData(myo::Myo *myo, uint64_t timestamp, const myo::Quaternion< float > &rotation) {
	int myoNumber = getMyoNumber(myo);
	if (myoNumber == 0)
	{
		orientationFile1 << timestamp
			<< ',' << rotation.x()
			<< ',' << rotation.y()
			<< ',' << rotation.z()
			<< ',' << rotation.w()
			<< std::endl;

		using std::atan2;
		using std::asin;
		using std::sqrt;
		using std::max;
		using std::min;

		// Calculate Euler angles (roll, pitch, and yaw) from the unit quaternion.
		float roll = atan2(2.0f * (rotation.w() * rotation.x() + rotation.y() * rotation.z()),
			1.0f - 2.0f * (rotation.x() * rotation.x() + rotation.y() * rotation.y()));
		
		float pitch = asin(max(-1.0f, min(1.0f, 2.0f * (rotation.w() * rotation.y() - rotation.z() * rotation.x()))));

		float yaw = atan2(2.0f * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
			1.0f - 2.0f * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));

		if (!calibrated_1)
		{
			roll_1 = roll;
			pitch_1 = pitch;
			yaw_1 = yaw;
			calibrated_1 = true;
		}
		
		//We make the values relative to the first position
		roll = roll - roll_1;
		pitch = pitch - pitch_1;
		yaw = yaw - yaw_1;

		orientationEulerFile1 << timestamp
			<< ',' << roll
			<< ',' << pitch
			<< ',' << yaw
			<< std::endl;
	}
	else if (myoNumber == 1)
	{
		orientationFile2 << timestamp
			<< ',' << rotation.x()
			<< ',' << rotation.y()
			<< ',' << rotation.z()
			<< ',' << rotation.w()
			<< std::endl;

		using std::atan2;
		using std::asin;
		using std::sqrt;
		using std::max;
		using std::min;

		// Calculate Euler angles (roll, pitch, and yaw) from the unit quaternion.
		float roll = atan2(2.0f * (rotation.w() * rotation.x() + rotation.y() * rotation.z()),
			1.0f - 2.0f * (rotation.x() * rotation.x() + rotation.y() * rotation.y()));
		float pitch = asin(max(-1.0f, min(1.0f, 2.0f * (rotation.w() * rotation.y() - rotation.z() * rotation.x()))));
		float yaw = atan2(2.0f * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
			1.0f - 2.0f * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));

		//We save the first position read for using it as the beginning
		if (!calibrated_2)
		{
			roll_2 = roll;
			pitch_2 = pitch;
			yaw_2 = yaw;
			calibrated_2 = true;
		}

		//We make the values relative to the first position
		roll = roll - roll_2;
		pitch = pitch - pitch_2;
		yaw = yaw - yaw_2;

		orientationEulerFile2 << timestamp
			<< ',' << roll
			<< ',' << pitch
			<< ',' << yaw
			<< std::endl;
	}
}
/*
// Compute a vector that points perpendicular to the forward direction,
// minimizing angular distance from world up (positive Y axis).
// This represents the direction of no rotation about its forward axis.
myo::Vector3<float> computeZeroRollVector(myo::Vector3<float> forward)
{
	myo::Vector3<float> *antigravity = new myo::Vector3<float>(0, 1, 0);
	myo::Vector3<float> m = forward.cross(*antigravity);
	myo::Vector3<float> roll = m.cross(forward);
	return roll.normalized;
}
*/
// onAccelerometerData is called whenever new acceleromenter data is provided
// Be warned: This will not make any distiction between data from other Myo armbands
void MyoDataCollector::onAccelerometerData(myo::Myo *myo, uint64_t timestamp, const myo::Vector3< float > &accel) {
	int myoNumber = getMyoNumber(myo);
	if (myoNumber == 0)
	{
		printVector(accelerometerFile1, timestamp, accel);

		/*
		if (!calibrated_1)
		{
			// _antiYaw represents a rotation of the Myo armband about the Y axis (up) which aligns the forward
			// vector of the rotation with Z = 1 when the wearer's arm is pointing in the reference direction.
			const myo::Vector3<float>* vector1 = new myo::Vector3<float>(forward_x, 0, forward_z);
			const myo::Vector3<float>* vector2 = new myo::Vector3<float>(0, 0, 1);
			_antiYaw = myo::rotate(
				*vector1,
				*vector2
			);
			
			// _referenceRoll represents how many degrees the Myo armband is rotated clockwise
			// about its forward axis (when looking down the wearer's arm towards their hand) from the reference zero
			// roll direction. This direction is calculated and explained below. When this reference is
			// taken, the joint will be rotated about its forward axis such that it faces upwards when
			// the roll value matches the reference.
			myo::Vector3<float>* vector3 = new myo::Vector3<float>(0, 0, 1);
			myo::Vector3<float>* vector4 = new myo::Vector3<float>(0, 1, 0);
			myo::Vector3<float> referenceZeroRoll = computeZeroRollVector(*vector3);
			_referenceRoll = rollFromZero(referenceZeroRoll, *vector3, *vector4);


			calibrated_1 = true;
		}*/
	}
	else if (myoNumber == 1)
	{
		printVector(accelerometerFile2, timestamp, accel);
	}

}

// Compute the angle of rotation clockwise about the forward axis relative to the provided zero roll direction.
// As the armband is rotated about the forward axis this value will change, regardless of which way the
// forward vector of the Myo is pointing. The returned value will be between -180 and 180 degrees.
float MyoDataCollector::rollFromZero(myo::Vector3<float> zeroRoll, myo::Vector3<float> forward, myo::Vector3<float> up)
{
	// The cosine of the angle between the up vector and the zero roll vector. Since both are
	// orthogonal to the forward vector, this tells us how far the Myo has been turned around the
	// forward axis relative to the zero roll vector, but we need to determine separately whether the
	// Myo has been rolled clockwise or counterclockwise.
	float cosine = up.dot(zeroRoll);

	// To determine the sign of the roll, we take the cross product of the up vector and the zero
	// roll vector. This cross product will either be the same or opposite direction as the forward
	// vector depending on whether up is clockwise or counter-clockwise from zero roll.
	// Thus the sign of the dot product of forward and it yields the sign of our roll value.
	myo::Vector3<float> cp = up.cross(zeroRoll);
	float directionCosine =forward.dot(cp);
	float sign = directionCosine < 0.0f ? 1.0f : -1.0f;

	// Return the angle of roll (in degrees) from the cosine and the sign.
	return sign * (360 / (3.14159265358979323 * 2)) * acos(cosine);
}


// onGyroscopeData is called whenever new gyroscope data is provided
// Be warned: This will not make any distiction between data from other Myo armbands
void MyoDataCollector::onGyroscopeData(myo::Myo *myo, uint64_t timestamp, const myo::Vector3< float > &gyro) {
	int myoNumber = getMyoNumber(myo);
	if (myoNumber == 0)
	{
		printVector(gyroFile1, timestamp, gyro);
	}
	else if (myoNumber == 1)
	{
		printVector(gyroFile2, timestamp, gyro);
	}

}

void MyoDataCollector::onConnect(myo::Myo *myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion) {
	//Reneable streaming
	myo->setStreamEmg(myo::Myo::streamEmgEnabled);


	/*if (!opened)
	{
		openFiles();
		opened = true;
	}*/
}

// Helper to print out accelerometer and gyroscope vectors
void MyoDataCollector::printVector(std::ofstream &file, uint64_t timestamp, const myo::Vector3< float > &vector) {
	file << timestamp
		<< ',' << vector.x()
		<< ',' << vector.y()
		<< ',' << vector.z()
		<< std::endl;
}

// Every time Myo Connect successfully pairs with a Myo armband, this function will be called.
//
// You can rely on the following rules:
//  - onPair() will only be called once for each Myo device
//  - no other events will occur involving a given Myo device before onPair() is called with it
//
// If you need to do some kind of per-Myo preparation before handling events, you can safely do it in onPair().
void  MyoDataCollector::onPair(myo::Myo* myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion)
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


void MyoDataCollector::registerNewMyo(myo::Myo* myo)
{
	if (std::find(knownMyos.begin(), knownMyos.end(), myo) != knownMyos.end())
	{
		std::cout << "This myo device had already been registerd\n";
	}
	else
	{
	    std::cout << "The myo device " << myo << " has been registered successfully\n";
		knownMyos.push_back(myo);
	}
}


int MyoDataCollector::getMyoNumber(myo::Myo* myo)
{
	std::vector <myo::Myo*>::iterator i = knownMyos.begin();


	i = find(knownMyos.begin(), knownMyos.end(), myo);

	if (i != knownMyos.end())
	{
		int nPosition = distance(knownMyos.begin(), i);
		return nPosition;
	}
	/*

	std::vector<myo::Myo*>::iterator iter = std::find_if(knownMyos.begin(), knownMyos.end(), myo);
	size_t index = std::distance(knownMyos.begin(), iter);
	if (index < knownMyos.size() && index >=0 )
	{
		int answer = index;
		return answer;
	}
	*/
}

// onUnpair() is called whenever the Myo is disconnected from Myo Connect by the user.
void   MyoDataCollector::onUnpair(myo::Myo* myo, uint64_t timestamp)
{
	// We've lost a Myo.
	// Let's clean up some leftover state.
	emgSamples1.fill(0);
	emgSamples2.fill(0);
}
/*
// onEmgData() is called whenever a paired Myo has provided new EMG data, and EMG streaming is enabled.
void MyoDataCollector::onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg)
{
	std::cout << "ando recibiendo de " << myo << "  ";
	
	if(knownMyos.size() > 0)
	{
		if (myo == knownMyos[0])
		{
			std::cout << "el uno";
			for (int i = 0; i < 8; i++) {
				emgSamples1[i] = emg[i];
			}
		}
		else if(knownMyos.size() > 1)
		{
		    if (myo == knownMyos[1])
			{
				std::cout << "el dos";
				for (int i = 0; i < 8; i++) {
					emgSamples2[i] = emg[i];
				}
			}
		}
	}
	
	std::cout << "\n";
	
	

}
*/
// There are other virtual functions in DeviceListener that we could override here, like onAccelerometerData().
// For this example, the functions overridden above are sufficient.

// We define this function to print the current values that were updated by the on...() functions above.
void MyoDataCollector::print()
{
	// Clear the current line
	std::cout << '\r';

	// Print out the EMG data.
	for (size_t i = 0; i < emgSamples1.size(); i++) {
		std::ostringstream oss;
		oss << static_cast<int>(emgSamples1[i]);
		std::string emgString = oss.str();

		std::cout << '[' << emgString << std::string(4 - emgString.size(), ' ') << ']';
	}

	std::cout << std::flush;
}




