#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
using namespace thrust;
using namespace thrust::placeholders;

// This example computes the number of words in a text sample. The algorithm counts the number of characters which start a new word, i.e. the number of characters where text[i] is an alphabetical character and text[i-1] is not an alphabetical character. determines whether the character is alphabetical
int main(int argc, char* argv[])
{
	// Paragraph from 'The Raven' by Edgar Allan Poe. http://en.wikipedia.org/wiki/The_Raven
	const char h[] =
	"  But the raven, sitting lonely on the placid bust, spoke only,\n"
	"  That one word, as if his soul in that one word he did outpour.\n"
	"  Nothing further then he uttered - not a feather then he fluttered -\n"
	"  Till I scarcely more than muttered `Other friends have flown before -\n"
	"  On the morrow he will leave me, as my hopes have flown before.'\n"
	"  Then the bird said, `Nevermore.'\n";
	
	// Copy the text from host to device.
	device_vector<char> d(h, h + sizeof(h));

	// Count words.
	int wc = inner_product
	(
		d.begin(), d.end() - 1, // Iterator of left characters.
		d.begin() + 1,          // Iterator of right characters.
		0,                      // Initial value of the reduction.
		plus<int>(),            // Binary operation used to reduce values.
		!('A' <= _1 && _1 <= 'z') && ('A' <= _2 && _2 <= 'z') // Functor to determine whether a new word starts.
	);
	
	// If the first character is alphabetical, then it also begins a word.
	if ('A' <= h[0] && h[0] <= 'z') ++wc;
	
	// Print.
	std::cout << wc << std::endl;
}
