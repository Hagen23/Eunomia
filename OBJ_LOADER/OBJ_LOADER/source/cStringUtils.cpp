#include "cStringUtils.h"

StringUtils::StringUtils()
{

}

StringUtils::~StringUtils()
{

}

string StringUtils::getBasePath(const string& path)
{
	size_t pos = path.find_last_of("\\/");
	return (string::npos == pos) ? "" : path.substr(0, pos + 1);
}

string StringUtils::getFileName(const string& path)
{
	size_t pos = path.find_last_of("\\/");
	return (string::npos == pos) ? path : path.substr(pos + 1);
}
