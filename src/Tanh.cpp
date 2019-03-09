#include "Tanh.h"
#include "config.h"

//e^x
//inline float exp2(float x) {
//	x = 1.0 + x / 1024;
//	x *= x; x *= x; x *= x; x *= x;
//	x *= x; x *= x; x *= x; x *= x;
//	x *= x; x *= x;
//	return x;
//}

Mat Tanh::Activation(Mat input) {
	int row = input.rows;
	int col = input.cols;
	Mat pos_exp_x(row, col, CV_32FC1);
	Mat neg_exp_x(row, col, CV_32FC1);
	Mat dest(row, col, CV_32FC1);
#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			pos_exp_x.at<float>(i, j) = exp(input.at<float>(i, j));
		}
	}

#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			neg_exp_x.at<float>(i, j) = exp(-input.at<float>(i, j));
		}
	}

#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dest.at<float>(i, j) = (pos_exp_x.at<float>(i, j) - neg_exp_x.at<float>(i, j)) / (pos_exp_x.at<float>(i, j) + neg_exp_x.at<float>(i, j));
		}
	}

	return dest;
}

Mat Tanh::DeActivation(Mat input) {
	int row = input.rows;
	int col = input.cols;
	Mat tanhx = Tanh::Activation(input);
	Mat tanhx2(row, col, CV_32FC1);
	Mat dest(row, col, CV_32FC1);

#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			tanhx2.at<float>(i, j) = tanhx.at<float>(i, j) * tanhx.at<float>(i, j);
		}
	}

#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dest.at<float>(i, j) = 1.0 - tanhx2.at<float>(i, j);
		}
	}

	return dest;
}