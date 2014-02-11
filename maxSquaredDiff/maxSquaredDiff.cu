#include <iostream>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
using namespace thrust;
using namespace thrust::placeholders;

int main(int argc, char* argv[])
{
	// Initialize two vectors.
    device_vector<float> a(4);
    device_vector<float> b(4);
    a[0] = 1.0;  b[0] = 2.0; 
    a[1] = 2.0;  b[1] = 4.0;
    a[2] = 3.0;  b[2] = 3.0;
    a[3] = 4.0;  b[3] = 0.0;

	// Compute the maximum squared difference.
    float max_squared_diff = inner_product
    (
    	a.begin(), a.end(),   // Data range 1.
    	b.begin(),            // Data range 2.
    	0,                    // Initial value for the reduction.
    	maximum<float>(),     // Binary operation used to reduce values.
		(_1 - _2) * (_1 - _2) // Lambda expression to compute squared difference.
    );

    //TODO:
    //    When no placeholder of _2 is passed, degrade to transform_reduce.
    //    An existing feature?
    //    This is for performance sake.

	// Print the result.
    std::cout << max_squared_diff << std::endl;
}
