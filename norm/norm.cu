#include <cmath>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include <time.h>

using namespace thrust;
using namespace thrust::placeholders;

#define NUM 100000000

int my_rand(void)
{
	static default_random_engine rng;
	static uniform_int_distribution<float> dist(0, 1000);
	return dist(rng);
}

int main(int argc, char* argv[])
{
	// manual data generation
	//// Initialize host data.
	//float h[4] = {1.0, 2.0, 3.0, 4.0};
	//// Copy data from host to device.
	//device_vector<float> d(h, h + 4);

	time_t start_time;

	start_time = clock();

	host_vector<float> h(NUM);
	generate(h.begin(), h.end(), my_rand);

	device_vector<float> d = h;

	std::cout << "gen data done: " << clock() - start_time << std::endl;

	start_time = clock();

	// Compute norm square.
	//http://thrust.github.io/doc/group__transformed__reductions.html#ga0d4232a9685675f488c3cc847111e48d
	float norm2 = transform_reduce
	(
		d.begin(), d.end(), // Data range.
		_1 * _1,            // Unary transform operation.
		0,                  // Initial value of the reduction.
		plus<float>()       // Binary operation used to reduce values.
	);
	// Compute norm.
	float norm = std::sqrt(norm2);
	// Print the norm.
	std::cout << norm << std::endl;

	std::cout << "first method:" << clock() - start_time << std::endl;

	start_time = clock();
	device_vector<float> tmp = d;
	transform(
		tmp.begin(), tmp.end(),
		tmp.begin(),
		// _1 is defined in placeholders
		_1 * _1
	);
	float norm2_new = reduce(tmp.begin(), tmp.end(), 0, plus<float>());
	float norm_new = std::sqrt(norm2_new);
	std::cout << norm_new << std::endl;
	std::cout << "second method:" << clock() - start_time << std::endl;
}
