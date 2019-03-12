#include "Power.h"
#include "config.h"


Mat Power::Activation(float shift, float scale, float power){
		int row = input.rows;
		int col = input.cols;
		Mat dest(row, col, CV_32FC1);

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dest.at<float>(i, j) = pow(shift + scale * input.at<float>(i, j), power);
			}
		}

		return dest;

	}
	

Mat Power::DeActivation(float shift , float scale , float power) {
	int row = input.rows;
	int col = input.cols;
	Mat dest(row, col, CV_32FC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dest.at<float>(i, j) = power * scale * pow(shift + scale * input.at<float>(i, j), power-1);
		}
	}

	return dest;
}