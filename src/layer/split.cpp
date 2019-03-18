#include "split.h"

vector <vector<Mat>> Split::Activation() {
	if (input.size() == 0) {
		fprintf(stderr, "PRelu Input Is Empty!");
		exit(-1);
	}
	vector <vector <Mat>> output;
	output.resize(copy_size);
	for (int i = 0; i < copy_size; i++) {
		output[i] = input;
	}
	return output;
}