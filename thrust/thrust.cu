#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/random.h>

int my_rand()
{
	static thrust::default_random_engine rng;
	static thrust::uniform_int_distribution<int> dist(0, 999);
	return dist(rng);
}

int main(int argc, char* argv[])
{
	// Allocate a vector in host memory.
	thrust::host_vector<int> h(1e+7);

	// Generate random numbers on the host.
	thrust::generate(h.begin(), h.end(), my_rand);

	// Copy the vector from host memory to device memory.
	thrust::device_vector<int> d = h;
 
	// Compute the sum on the device.
	int actual = thrust::reduce(d.begin(), d.end(), 0, thrust::plus<int>());

	// Compute the sum on the host.
	int expected = 0;
	for (int i = 0; i < h.size(); ++i)
	{
		expected += h[i];
	}

	// Validate the result.
	if (actual != expected)
	{
		std::cout << "actual = " << actual << ", expected = " << expected << std::endl;
	}
}
