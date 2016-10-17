CC = g++
CFLAGS =-c -std=c++11 
INCLUDES = -I../common
LDFLAGS = -lGL -lglut -lGLU
SOURCES = Lattice_3D.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
OUTF = ../build/3d_free_surface

build: $(SOURCES) $(OUTF)

$(OUTF): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# $@
# The file name of the target of the rule. If the target is an archive member, then ‘$@’ is the name of the archive file. In a pattern rule that has multiple targets (see Introduction to Pattern Rules), ‘$@’ is the name of whichever target caused the rule’s recipe to be run.

# $<
# The name of the first prerequisite. If the target got its recipe from an implicit rule, this will be the first prerequisite added by the implicit rule (see Implicit Rules).

# A pattern rule: in order to build a file ending in .o, first rebuild/create/find the corresponding file ending in .cpp, and then run the command $(CC) $(CFLAGS) -o $@ $<.

clean:
	rm *.o 

rebuild: clean build
