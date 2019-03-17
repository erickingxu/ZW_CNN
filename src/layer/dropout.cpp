#include "dropout.h"

vector <Mat> Dropout::Activation() {
	if (input.empty()) {
		fprintf(stderr, "Dropout Input Is Empty!");
		exit(-1);
	}
	vector <Mat> output = input;
	return output;
}