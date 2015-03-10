#include "cTextIO.h"

TextIO::TextIO()
{
}

TextIO::~TextIO()
{
}

string TextIO::textFileRead(string _fname)
{
	cout << "READING FILE '" << _fname.c_str() << "'" << endl;
	string result = "";
	string line;
	ifstream myfile(_fname);
	if( myfile.is_open() )
	{
		while( getline(myfile, line) )
		{
			result.append(line);
			result.append("\n");
#ifdef __PRINT_TEXT
			cout << line << endl;
#endif
		}
		myfile.close();
		cout << "DONE." << endl;
	}
	else
	{
		cout << "ERROR. UNABLE TO OPEN FILE '" << _fname.c_str() << "'" << endl;
	}
	return result;
}

bool TextIO::textFileWrite(string _fname, string _s)
{
	cout << "WRITING FILE '" << _fname.c_str() << "'" << endl;
	ofstream myfile(_fname);
	if( myfile.is_open() )
	{
#ifdef __PRINT_TEXT
		cout << _s.c_str() << endl;
#endif
		myfile << _s.c_str();
		myfile.close();
		cout << "DONE." << endl;
		return true;
	}
	else
	{
		cout << "ERROR. UNABLE TO OPEN FILE '" << _fname.c_str() << "'" << endl;
		return false;
	}
}
