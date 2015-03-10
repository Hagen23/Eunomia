#pragma once
#include <map>
#include <iostream>

#include "il.h"
#include "glew.h"
#include "Scene.h"

#include "cStringUtils.h"

using namespace std;

#ifndef __TEXTURE_MANAGER
#define __TEXTURE_MANAGER

class TextureManager
{
public:
	TextureManager(string _basepath);
	~TextureManager();

	bool LoadGLTextures(const aiScene* scene);
	// images / texture
	// map image filenames to textureIds
	// pointer to texture Array
	map<string, GLuint> textureIdMap;

private:
	string basepath;
};

#endif /*__TEXTURE_MANAGER*/
