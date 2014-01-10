#include <iostream>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
using namespace thrust;

struct functor
{
	template <typename Tuple>
	__device__
	void operator()(Tuple t)
	{
		// D[i] = A[i] + B[i] * C[i];
		get<3>(t) = get<0>(t) + get<1>(t) * get<2>(t);
		// E[i] = A[i] + B[i] + C[i];
		get<4>(t) = get<0>(t) + get<1>(t) + get<2>(t);
	}
};

int main(int argc, char* argv[])
{
	// Initialize input vectors
	device_vector<float> A(5);
	device_vector<float> B(5);
	device_vector<float> C(5);
	device_vector<float> D(5);
	device_vector<float> E(5);
	A[0] = 3;  B[0] = 6;  C[0] = 2; 
	A[1] = 4;  B[1] = 7;  C[1] = 5; 
	A[2] = 0;  B[2] = 2;  C[2] = 7; 
	A[3] = 8;  B[3] = 1;  C[3] = 4; 
	A[4] = 2;  B[4] = 8;  C[4] = 3; 

	// Apply the transformation.
	for_each
	(
		make_zip_iterator
		(
			make_tuple
			(
				A.begin(),
				B.begin(),
				C.begin(),
				D.begin(),
				E.begin()
			)
		),
		make_zip_iterator
		(
			make_tuple
			(
				A.end(),
				B.end(),
				C.end(),
				D.end(),
				E.end()
			)
		),
		functor()
	);

	// Print the output.
	for (int i = 0; i < 5; ++i)
	{
		std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl
		          << A[i] << " + " << B[i] << " + " << C[i] << " = " << E[i] << std::endl;
	}
}
