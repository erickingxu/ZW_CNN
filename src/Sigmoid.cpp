#include "Sigmoid.h"
#include "Network.h"

//e^x
inline float exp2(float x) {
	x = 1.0 + x / 1024;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x;
	return x;
}

Mat Sigmoid::Activation(Mat input) {
	int row = input.rows;
	int col = input.cols;
	Mat dest(row, col, CV_32FC1);
	Mat exp_x(row, col, CV_32FC1);
#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			exp_x.at<float>(i, j) = exp2(-input.at<float>(i, j));
		}
	}

#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dest.at<float>(i, j) = 1.0 / (1.0 + exp_x.at<float>(i, j));
		}
	}

	return dest;
}

Mat Sigmoid::DeActivation(Mat input) {
	int row = input.rows;
	int col = input.cols;
	Mat psigmoidx = Sigmoid::Activation(input);
	Mat nsigmoidx(row, col, CV_32FC1);
	Mat dest(row, col, CV_32FC1);

#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			nsigmoidx.at<float>(i, j) = 1.0 - psigmoidx.at<float>(i, j);
		}
	}

#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dest.at<float>(i, j) = psigmoidx.at<float>(i, j) * nsigmoidx.at<float>(i, j);
		}
	}

	return dest;
}