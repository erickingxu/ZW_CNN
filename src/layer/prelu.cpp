#include "prelu.h"

vector <Mat> PRelu::Activation() {
	int Size = input.size();
	if (input.size() == 0) {
		fprintf(stderr, "PRelu Input Is Empty!");
		exit(-1);
	}
	vector <Mat> output;
	output.resize(Size);
	for (int i = 0; i < Size; i++) {
		Mat dest(input[i].rows, input[i].cols, CV_32FC1);
		for (int j = 0; j < input[i].rows; j++) {
			for (int k = 0; k < input[i].cols; k++) {
				if (input[i].at<float>(j, k) > 0) {
					dest.at<float>(j, k) = input[i].at<float>(j, k);
				}
				else {
					dest.at <float>(j, k) = alpha * input[i].at<float>(j, k);
				}
			}
		}
		output[i] = dest;
	}
	return output;
}