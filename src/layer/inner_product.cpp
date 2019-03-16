#include "inner_product.h"

Mat Inner_Product::Activation() {
	if (input.size() != weights.size()) {
		fprintf(stderr, "Inner_Product Dims Don't Match!");
		exit(-1);
	}
	Mat output = input * weights;
	return output;
}