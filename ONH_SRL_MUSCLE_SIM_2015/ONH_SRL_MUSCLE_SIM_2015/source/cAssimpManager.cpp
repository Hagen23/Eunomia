#include "cAssimpManager.h"

AssimpManager::AssimpManager(string _mpath)
{
	scene = NULL;
	model_path = _mpath;
	scaleFactor = 1.0f;
}

AssimpManager::~AssimpManager()
{
}

void AssimpManager::get_bounding_box_for_node(const aiNode* nd, aiVector3D* min, aiVector3D* max)
{
	aiMatrix4x4 prev;
	unsigned int n = 0, t;

	for (; n < nd->mNumMeshes; ++n)
	{
		const aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
		for (t = 0; t < mesh->mNumVertices; ++t)
		{
			aiVector3D tmp = mesh->mVertices[t];
			min->x = aisgl_min(min->x, tmp.x);
			min->y = aisgl_min(min->y, tmp.y);
			min->z = aisgl_min(min->z, tmp.z);

			max->x = aisgl_max(max->x, tmp.x);
			max->y = aisgl_max(max->y, tmp.y);
			max->z = aisgl_max(max->z, tmp.z);
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n)
	{
		get_bounding_box_for_node(nd->mChildren[n], min, max);
	}
}

void AssimpManager::get_bounding_box(void)
{
	scene_min.x = scene_min.y = scene_min.z = 1e10f;
	scene_max.x = scene_max.y = scene_max.z = -1e10f;
	get_bounding_box_for_node(scene->mRootNode, &scene_min, &scene_max);
}

bool AssimpManager::Import3DFromFile(void)
{
	//check if file exists
	std::ifstream fin(model_path.c_str());
	if (!fin.fail())
	{
		fin.close();
	}
	else
	{
		qDebug() << "\nERROR. COULD NOT OPEN FILE: '" << model_path.c_str() << "'. MESSAGE: '" << importer.GetErrorString() << "'\n";
		return false;
	}
	scene = importer.ReadFile(model_path, aiProcessPreset_TargetRealtime_Quality);
	// If the import failed, report it
	if (!scene)
	{
		qDebug() << "\nERROR IMPORTING SCENE. MESSAGE: '" << importer.GetErrorString() << "'\n";
		return false;
	}
	// Now we can access the file's contents.
	qDebug() << "\nIMPORT OF SCENE '" << model_path.c_str() << "' SUCCESSFUL.\n";

	get_bounding_box();
	float tmp;
	tmp = scene_max.x - scene_min.x;
	tmp = scene_max.y - scene_min.y > tmp ? scene_max.y - scene_min.y : tmp;
	tmp = scene_max.z - scene_min.z > tmp ? scene_max.z - scene_min.z : tmp;
	scaleFactor = 1.f / tmp;

	// We're done. Everything will be cleaned up by the importer destructor
	return true;
}

const aiScene* AssimpManager::getScene(void)
{
	return scene;
}

float AssimpManager::getScaleFactor(void)
{
	return scaleFactor;
}
