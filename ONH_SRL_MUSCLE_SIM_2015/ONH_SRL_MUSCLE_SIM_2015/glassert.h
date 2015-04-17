#ifndef GL_ASSERT_H
#define GL_ASSERT_H

/**
* OpenGL error management class.
*/

#ifndef NDEBUG // debug mode

#include <iostream>
#include <cassert>

#ifndef __TO_STR
#define __TO_STR(x) __EVAL_STR(x)
#define __EVAL_STR(x) #x
#endif


#define glAssert(code) \
    code; \
	    {\
        GLuint err = glGetError(); \
        if (err != GL_NO_ERROR) { \
            qDebug() <<"\nOpenGL error("<<__FILE__<<":"<<__LINE__<<", "<<__TO_STR(code)<<") : "<<"code("<<err<<")\n"; \
            assert(false); \
		        } \
	    }


#define glCheckError() \
	    {\
        GLuint err = glGetError(); \
        if (err != GL_NO_ERROR) { \
            qDebug()<<"\nOpenGL error("<<__FILE__<<":"<<__LINE__<<") : "<<"code("<<err<<")\n"; \
            assert(false); \
		        } \
	    }

#else // No debug
#define glAssert(code) code;
#define glCheckError()
#endif


#endif // GL_ASSERT_H
