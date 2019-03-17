#include "concat.h"

vector<Mat> Concat::Activation() {
	int Size = input.size();
	for (int i = 1; i < Size; i++) {
		if (input[i].size() != input[0].size()) {
			fprintf(stderr, "Input Dims Dont Match!");
			exit(-1);
		}
	}
	vector <Mat> output;
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < input[i].size(); j++) {
			output.push_back(input[i][j]);
		}
	}
	return output;
}