#include "slice.h"

vector<vector<Mat> > Slice::Activation() {
	int Size1 = input.size();
	int Size2 = slice_point.size();
	if (Size1 - 1 != Size2) {
		fprintf(stderr, "Input Dims Dont Match!");
		exit(-1);
	}
	vector <vector <Mat>> output;
	output.resize(Size1);
	int last = 0;
	for (int i = 0; i < Size1; i++) {
		for (int j = last; j <= i; j++) {
			output[i].push_back(input[j]);
		}
	}
	return output;
}