//Custom objects:
AssimpManager*		assimp_manager;
TextureManager*		tex_manager;
Shader*				shader;
vector<AssimpMesh>	myMeshes;

// Model Matrix (part of the OpenGL Model View Matrix)
float modelMatrix[16];
// For push and pop matrix
vector<float*> matrixStack;

// Uniform Buffer for Matrices
// this buffer will contain 3 matrices: projection, view and model
// each matrix is a float array with 16 components
GLuint matricesUniBuffer;

// Shader Names
char *vertexFileName = "shaders/dirlightdiffambpix.vert";
char *fragmentFileName = "shaders/dirlightdiffambpix.frag";

// Replace the model name by your model's filename
//static const string modelname = "assets/bench.obj";
//static const string modelname = "assets/sphere.obj";
//static const string modelname = "assets/sphere_notex.obj";
//static const string modelname = "assets/spider.obj";
//static const string modelname = "assets/testmixed.obj";
//static const string modelname = "assets/WusonOBJ.obj";
static const string modelname = "assets/arm.obj";

// Camera Position
float camX = 0, camY = 0, camZ = 5;

// Mouse Tracking Variables
int startX, startY, tracking = 0;

// Camera Spherical Coordinates
float alpha = 0.0f, beta = 0.0f;
float r = 5.0f;

// Frame counting and FPS computation
long time, timebase = 0, frame = 0;
char s[32];
