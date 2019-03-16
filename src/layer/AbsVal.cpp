#include "absval.h"

vector <Mat> AbsVal::Activation() {
	int Size = input.size();
	int row = input[0].rows;
	int col = input[1].cols;
	vector <Mat> output;
	output.resize(Size);
	for (int i = 0; i < Size; i++) {
		Mat dest(row, col, CV_32FC1);
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				dest.at<float>(j, k) = input[i].at<float>(j, k) >= 0 ? input[i].at<float>(j, k) : -(input[i].at<float>(j, k));
			}
		}
		output[i] = dest;
	}
	return output;
}