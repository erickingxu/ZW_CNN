#include "argmax.h"

int ArgMax::Activation() {
	int row = input.rows;
	int col = input.cols;
	if (col != 1) {
		fprintf(stderr, "ArgMax Input Dims Don't Match");
		exit(-1);
	}
	int final_index = 0;
	int maxx = input.at<float>(0, 0);
	for (int i = 1; i < row; i++) {
		if (input.at<float>(i, 0) > maxx) {
			maxx = input.at<float>(i, 0);
			final_index = i;
		}
	}
	return final_index;
}