#pragma once
#include <string>

using namespace std;

#ifndef __STRING_UTILS
#define __STRING_UTILS

class StringUtils
{
public:
	StringUtils();
	~StringUtils();

	static string getBasePath(const string& path);
	static string getFileName(const string& path);
};

#endif /*__STRING_UTILS*/
