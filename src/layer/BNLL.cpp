#include "BNLL.h"
#include "config.h"

Mat BNLL::Activation(){

	int row = input.rows;
	int col = input.cols;
	Mat dest(row, col, CV_32FC1);
	Mat exp_x(row, col, CV_32FC1);
	//#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			exp_x.at<float>(i, j) = exp(input.at<float>(i, j));
		}
	}

	//#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dest.at<float>(i, j) =  log(1.0 + exp_x.at<float>(i, j));
		}
	}

	return dest;
}

