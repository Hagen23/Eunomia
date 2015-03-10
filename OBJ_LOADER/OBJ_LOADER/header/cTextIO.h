#pragma once

#include <string>
#include <iostream>
#include <fstream>

using namespace std;

#undef __PRINT_TEXT

#ifndef __TEXTIO
#define __TEXTIO

class TextIO
{
public:
	TextIO();
	~TextIO();

	string textFileRead(string _fname);
	bool textFileWrite(string _fname, string _s);
};

#endif /*TEXTIO*/
