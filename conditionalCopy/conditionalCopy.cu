#include <thrust/iterator/counting_iterator.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
using namespace thrust;

int main(int argc, char* argv[])
{
	// Sequence of zero and nonzero values.
	int h[] = {0, 1, 1, 0, 0, 1, 0, 1};

	// Copy data from host to device.
	device_vector<int> d(h, h + 8);

	// Storage for the nonzero indices.
	device_vector<int> indices(8);

	//// Compute indices of nonzero elements.
	//// http://thrust.github.io/doc/group__stream__compaction.html#ga36d9d6ed8e17b442c1fd8dc40bd515d5
	//device_vector<int>::iterator indices_end = copy_if
	//(
	//	counting_iterator<int>(0),
	//	counting_iterator<int>(8),
	//	d.begin(),
	//	indices.begin(),
	//	identity<int>()
	//);
	//// Print the indices.
	//thrust::copy(indices.begin(), indices_end, std::ostream_iterator<int>(std::cout, "\n"));

	int c = count_if
	(
		d.begin(),
		d.end(),
		identity<int>()
	);
	std::cout << c << std::endl;
}
