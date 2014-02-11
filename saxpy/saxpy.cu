#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
using namespace thrust;
using namespace thrust::placeholders;

#define ARRAY_LENGTH(x) sizeof(x) / sizeof(x[0])

int main(int argc, char* argv[])
{
	float a = 1.0f;
	float x[] = {1, 2, 3, 4, 7, 2};
	float y[] = {1, 1, 1, 1, 8, 9};
	int length = ARRAY_LENGTH(x); 

	device_vector<float> X(x, x + length);
	device_vector<float> Y(y, y + length);
	transform
	(
		X.begin(), X.end(), // Input range 1
		Y.begin(),          // Input range 2
		Y.begin(),          // Output range
		a * _1 + _2         // Lambda expression to compute ax + y
	);

	for (size_t i = 0; i < length; ++i)
	{
		std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;
	}
}
