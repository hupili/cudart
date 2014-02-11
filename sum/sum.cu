#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
using namespace thrust;

int my_rand(void)
{
	static default_random_engine rng;
	static uniform_int_distribution<int> dist(0, 9999);
	return dist(rng);
}

int main(int argc, char* argv[])
{
	//// Generate random data on the host.
	//host_vector<int> h(100);
	//generate(h.begin(), h.end(), my_rand);

	int h[] = {2, 2, 3}; 

	// Copy data from host to device.
	device_vector<int> d(h, h+3);

	// Compute sum on the device.
	int sum = reduce
	(
		d.begin(), d.end(), // Data range.
		0xff,                  // Initial value of the reduction.

		//plus<int>()         // Binary operation used to reduce values.

		//http://thrust.github.io/doc/structthrust_1_1multiplies.html
		// plus is called plus
		// but multiply is called multiplies
		//multiplies<int>()         // Binary operation used to reduce values.

		bit_and<int>()         // Binary operation used to reduce values.
	);

	// Print the sum.
	std::cout << sum << std::endl;
}
