#include "m3Matrix.h"

void m3Matrix::jacobiRotate(m3Matrix& A, m3Matrix& R, int p, int q)
{
	// rotates A through phi in pq-plane to set A(p,q) = 0
	// rotation stored in R whose columns are eigenvectors of A
	float d = (A(p, p) - A(q, q)) / (2.0f*A(p, q));
	float t = 1.0f / (fabs(d) + sqrt(d*d + 1.0f));
	if (d < 0.0f) t = -t;
	float c = 1.0f / sqrt(t*t + 1);
	float s = t*c;
	A(p, p) += t*A(p, q);
	A(q, q) -= t*A(p, q);
	A(p, q) = A(q, p) = 0.0f;
	// transform A
	int k;
	for (k = 0; k < 3; k++) 
	{
		if (k != p && k != q) 
		{
			float Akp = c*A(k, p) + s*A(k, q);
			float Akq = -s*A(k, p) + c*A(k, q);
			A(k, p) = A(p, k) = Akp;
			A(k, q) = A(q, k) = Akq;
		}
	}
	// store rotation in R
	for (k = 0; k < 3; k++) 
	{
		float Rkp = c*R(k, p) + s*R(k, q);
		float Rkq = -s*R(k, p) + c*R(k, q);
		R(k, p) = Rkp;
		R(k, q) = Rkq;
	}
}

//---------------------------------------------------------------------
void m3Matrix::eigenDecomposition(m3Matrix &A, m3Matrix &R)
//---------------------------------------------------------------------
{
	// only for symmetric matrices!
	// A = R A' R^T, where A' is diagonal and R orthonormal

	R.id();	// unit matrix
	int iter = 0;
	while (iter < JACOBI_ITERATIONS) 
	{	// 10 off diagonal elements
		// find off diagonal element with maximum modulus
		int p, q;
		float a, max;
		max = -1.0f;
		for (int i = 0; i < 2; i++)
		{
			for (int j = i + 1; j < 3; j++) 
			{
				a = fabs(A(i, j));
				if (max < 0.0f || a > max)
				{
					p = i; q = j; max = a;
				}
			}
		}
		// all small enough -> done
		//		if (max < EPSILON) break;  debug
		if (max <= 0.0f) break;
		// rotate matrix with respect to that element
		jacobiRotate(A, R, p, q);
		iter++;
	}
}

// --------------------------------------------------
void m3Matrix::polarDecomposition(const m3Matrix &A, m3Matrix &R, m3Matrix &S)
{
	// A = RS, where S is symmetric and R is orthonormal
	// -> S = (A^T A)^(1/2)

	R.id();	// default answer

	m3Matrix ATA;
	ATA.multiplyTransposedLeft(A, A);

	m3Matrix U;
	R.id();

	//Get eigenvalues; stored in ATA. This is in order to diagonalize the matrix.
	eigenDecomposition(ATA, U);

	//Obtain the U matrix by calculating the square root
	float l0 = ATA(0, 0); if (l0 <= 0.0f) l0 = 0.0f; else l0 = 1.0f / sqrt(l0);
	float l1 = ATA(1, 1); if (l1 <= 0.0f) l1 = 0.0f; else l1 = 1.0f / sqrt(l1);
	float l2 = ATA(2, 2); if (l2 <= 0.0f) l2 = 0.0f; else l2 = 1.0f / sqrt(l2);

	//And then calculate the inverse of U
	m3Matrix S1;
	S1.r00 = l0*U.r00*U.r00 + l1*U.r01*U.r01 + l2*U.r02*U.r02;
	S1.r01 = l0*U.r00*U.r10 + l1*U.r01*U.r11 + l2*U.r02*U.r12;
	S1.r02 = l0*U.r00*U.r20 + l1*U.r01*U.r21 + l2*U.r02*U.r22;

	S1.r10 = l0*U.r10*U.r00 + l1*U.r11*U.r01 + l2*U.r12*U.r02;
	S1.r11 = l0*U.r10*U.r10 + l1*U.r11*U.r11 + l2*U.r12*U.r12;
	S1.r12 = l0*U.r10*U.r20 + l1*U.r11*U.r21 + l2*U.r12*U.r22;

	S1.r20 = l0*U.r20*U.r00 + l1*U.r21*U.r01 + l2*U.r22*U.r02;
	S1.r21 = l0*U.r20*U.r10 + l1*U.r21*U.r11 + l2*U.r22*U.r12;
	S1.r22 = l0*U.r20*U.r20 + l1*U.r21*U.r21 + l2*U.r22*U.r22;

	/*S1.r10 = S1.r01;
	S1.r11 = l0*U.r10*U.r10 + l1*U.r11*U.r11*/;

	R.multiply(A, S1);
	S.multiplyTransposedLeft(R, A);
}