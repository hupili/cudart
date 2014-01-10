#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
using namespace thrust;
using namespace thrust::placeholders;

// This sample demonstrates the use of placeholders to implement the SAXPY operation Y[i] = a * X[i] + Y[i] where X and Y are vectors and a is a scalar constant.
int main(int argc, char* argv[])
{
	float a = 2.0f;
	float x[4] = {1, 2, 3, 4};
	float y[4] = {1, 1, 1, 1};

	device_vector<float> X(x, x + 4);
	device_vector<float> Y(y, y + 4);
	transform
	(
		X.begin(), X.end(), // input range 1
		Y.begin(),          // input range 2
		Y.begin(),          // output range
		a * _1 + _2         // placeholder expression
	);

	for (size_t i = 0; i < 4; ++i)
	{
		std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;
	}
}
