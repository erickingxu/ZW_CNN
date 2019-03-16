#include "tanh.h"
#include "config.h"

//e^x
//inline float exp2(float x) {
//	x = 1.0 + x / 1024;
//	x *= x; x *= x; x *= x; x *= x;
//	x *= x; x *= x; x *= x; x *= x;
//	x *= x; x *= x;
//	return x;
//}

vector <Mat> Tanh::Activation() {
	int Size = input.size();
	int row = input[0].rows;
	int col = input[0].cols;
	vector <Mat> output;
	output.resize(Size);

	for (int i = 0; i < Size; i++) {
		Mat pos_exp_x(row, col, CV_32FC1);
		Mat neg_exp_x(row, col, CV_32FC1);
		Mat dest(row, col, CV_32FC1);
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				pos_exp_x.at<float>(j, k) = exp(input[i].at<float>(j, k));
			}
		}
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				neg_exp_x.at<float>(j, k) = exp(-input[i].at<float>(j, k));
			}
		}
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				dest.at<float>(j, k) = (pos_exp_x.at<float>(j, k) - neg_exp_x.at<float>(j, k)) / (pos_exp_x.at<float>(j, k) + neg_exp_x.at<float>(j, k));
			}
		}
		output.push_back(dest);
	}
	return output;
}
