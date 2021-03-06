#pragma once
#include "glew.h"

#include <iostream>
#include <fstream>
#include <string>

#include "Importer.hpp"
#include "PostProcess.h"
#include "Scene.h"

#include "cMacros.h"

using namespace std;

#ifndef __ASSIMP_MANAGER
#define __ASSIMP_MANAGER

class AssimpManager
{
public:
	AssimpManager(string _mpath);
	~AssimpManager();

	bool Import3DFromFile(void);
	const aiScene* getScene(void);
	float getScaleFactor(void);

private:

	void get_bounding_box(void);
	void get_bounding_box_for_node(const aiNode* nd, aiVector3D* min, aiVector3D* max);

	Assimp::Importer importer;
	const aiScene* scene;
	aiVector3D scene_min;
	aiVector3D scene_max;
	string model_path;
	// scale factor for the model to fit in the window
	float scaleFactor;
};

#endif /*ASSIMP_MANAGER*/
