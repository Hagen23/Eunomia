#pragma once
#ifndef DEFORMABLE
#define DEFORMABLE

#include <vector>
#include "../Math3D/math3d.h"
#include "../Math3D/m3Bounds.h"
#include "../Math3D/m3Matrix.h"
#include "../Math3D/m9Matrix.h"

struct DeformableParameters
{
	DeformableParameters() { setDefaults(); }
	float timeStep;
	m3Vector gravity;
	m3Bounds bounds;

	float alpha;
	float beta;

	bool quadraticMatch;
	bool volumeConservation;

	bool allowFlip;

	void setDefaults();
};

class Deformable
{
public:
	Deformable();
	~Deformable();

	void reset();
	void addVertex(const m3Vector &pos, float mass);

	void externalForces();
	void projectPositions();
	void integrate();

	void timeStep();

	DeformableParameters params;

	int  getNumVertices() const { return mNumVertices; }
	const m3Vector & getVertexPos(int nr) { return mPos[nr]; }
	const m3Vector & getOriginalVertexPos(int nr) { return mOriginalPos[nr]; }
	const m3Vector & getGoalVertexPos(int nr) { return mGoalPos[nr]; }
	const m3Real getMass(int nr) { return mMasses[nr]; }

	void fixVertex(int nr, const m3Vector &pos);
	bool isFixed(int nr) { return mFixed[nr]; }
	void releaseVertex(int nr);

	void saveToFile(char *filename);
	void loadFromFile(char *filename);

private:
	void initState();

	int mNumVertices;
	std::vector<m3Vector> mOriginalPos;
	std::vector<m3Vector> mPos;
	std::vector<m3Vector> mNewPos;
	std::vector<m3Vector> mGoalPos;
	std::vector<float> mMasses;
	std::vector<m3Vector> mVelocities;
	std::vector<bool> mFixed;
};
#endif