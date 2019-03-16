#include "flatten.h"

Mat Flatten::Activation() {
	if (input.size() == 0) {
		fprintf(stderr, "Input Is Empty!");
		exit(-1);
	}
	int Size = input.size();
	int row = input[0].rows;
	int col = input[0].cols;
	Mat output(Size * row * col, 1, CV_32FC1);
	int index = 0;
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				output.at<float>(index, 0) = input[i].at<float>(i, j);
				index++;
			}
		}
	}
	return output;
}