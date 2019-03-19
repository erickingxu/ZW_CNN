#include "scale.h"
#include "config.h"

vector <Mat> Scale::Activation() {
	int Size = input.size();
	int row = input[0].rows;
	int col = input[0].cols;
	vector <Mat> output;
	output.resize(Size);
	for (int i = 0; i < Size; i++) {
		Mat dest(row, col, CV_32FC1);
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				dest.at<float>(j, k) = gamma * input[i].at<float>(j, k) + beta;
			}
		}
		output[i] = dest;
	}
	return output;
}