#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
using namespace thrust;
using namespace thrust::placeholders;

int main(int argc, char* argv[])
{
	float a = 2.0f;
	float x[4] = {1, 2, 3, 4};
	float y[4] = {1, 1, 1, 1};

	device_vector<float> X(x, x + 4);
	device_vector<float> Y(y, y + 4);
	transform
	(
		X.begin(), X.end(), // Input range 1
		Y.begin(),          // Input range 2
		Y.begin(),          // Output range
		a * _1 + _2         // Lambda expression to compute ax + y
	);

	for (size_t i = 0; i < 4; ++i)
	{
		std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;
	}
}
