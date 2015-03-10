#include "cTextureManager.h"

TextureManager::TextureManager(string _basepath)
{
	// initialization of DevIL:
	ilInit();
	basepath = _basepath;
}

TextureManager::~TextureManager()
{
	cout << "CLEARING TEXTURE ID MAP..." << endl;
	textureIdMap.clear();
	cout << "DONE." << endl;
}

bool TextureManager::LoadGLTextures(const aiScene* scene)
{
	ILboolean success;
	// Scan scene's materials for textures:
	for (unsigned int m = 0; m<scene->mNumMaterials; ++m)
	{
		int texIndex = 0;
		aiString path;	// filename

		aiReturn texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		while (texFound == AI_SUCCESS)
		{
			// Fill map with textures, OpenGL image ids set to 0:
			textureIdMap[path.data] = 0;
			// more textures?
			texIndex++;
			texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		}
	}

	int numTextures = textureIdMap.size();
	// Create and fill array with DevIL texture ids:
	ILuint* imageIds = new ILuint[numTextures];
	ilGenImages(numTextures, imageIds);

	// Create and fill array with GL texture ids:
	GLuint* textureIds = new GLuint[numTextures];
	glGenTextures(numTextures, textureIds); // Texture name generation

	// Get iterator:
	std::map<string, GLuint>::iterator itr = textureIdMap.begin();
	int i = 0;
	for (; itr != textureIdMap.end(); ++i, ++itr)
	{
		// Save IL image ID:
		string filename = (*itr).first;  // get filename
		filename = StringUtils::getFileName(filename);
		(*itr).second = textureIds[i];	  // save texture id for filename in map

		ilBindImage(imageIds[i]); // Binding of DevIL image name
		string fileloc = basepath + filename;	// Loading of image
		ilEnable(IL_ORIGIN_SET);
		ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
		success = ilLoadImage((ILstring)fileloc.c_str());

		if (success)
		{
			// Convert image to RGBA:
			ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);

			// Create and load textures to OpenGL:
			glBindTexture(GL_TEXTURE_2D, textureIds[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ilGetInteger(IL_IMAGE_WIDTH),
				ilGetInteger(IL_IMAGE_HEIGHT), 0, GL_RGBA, GL_UNSIGNED_BYTE,
				ilGetData());
		}
		else
		{
			cout << "ERROR. COULD NOT LOAD IMAGE '" << filename.c_str() << "'" << endl;
			return false;
		}
	}
	// Because we have already copied image data into texture data
	// we can release memory used by image.
	ilDeleteImages(numTextures, imageIds);
	// Cleanup
	delete[] imageIds;
	delete[] textureIds;
	// Return success;
	return true;
}
