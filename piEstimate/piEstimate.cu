#include <iostream>
#include <cmath>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
using namespace thrust;

// This example demonstrates a slightly more sophisticated technique
// which ensures that the subsequences generated in each thread are
// disjoint. To achieve this, we use a single common stream
// of random numbers, but partition it among threads to ensure no overlap
// of substreams. The substreams are generated procedurally using
// default_random_engine's discard(n) member function, which skips
// past n states of the RNG. This function is accelerated and executes
// in O(lg n) time.

struct estimate_pi : public unary_function<unsigned int,float>
{
	__device__
	float operator()(unsigned int t)
	{
		int N = 5000; // Number of samples per stream. Note that M * N <= default_random_engine::max, which is also the period of this particular RNG, ensures the substreams are disjoint.

		// Create a random number generator. Note that each thread uses an RNG with the same seed.
		default_random_engine rng;

		// Jump past the numbers used by the previous substreams. The time complexity is O(log N).
		rng.discard(N * t);

		// Create a mapping from random numbers to [0,1).
		uniform_real_distribution<float> u01(0,1);

		// Take N samples in a quarter circle.
		int sum = 0;
		for (int i = 0; i < N; ++i)
		{
			// Draw a sample from the unit square.
			float x = u01(rng);
			float y = u01(rng);

			// Measure distance from the origin.
			float dist = std::sqrt(x*x + y*y);

			// Count if (x,y) is inside the quarter circle.
			sum += dist <= 1.0f;
		}

		// Multiply by 4 to get the area of the whole circle, and normalize by N.
		return sum * 4.0f / N;
	}
};

int main(int argc, char* argv[])
{
	// Estimate pi for M times.
	int M = 30000;

	// Compute the sum of M estimations of pi.
	float pi_M = transform_reduce
	(
		counting_iterator<int>(0),
		counting_iterator<int>(M),
		estimate_pi(),
		0.0f,
		plus<float>()
	);

	// Compute pi.
	float pi = pi_M / M;

	// Print pi.
	std::cout << pi << std::endl;
}
