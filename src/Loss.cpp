#include "Loss.h"
#include "config.h"

float Loss::L1(Mat input, Mat label, Mat &out_error) {
	if (input.rows != label.rows || input.cols != label.cols) {
		fprintf(stderr, "Foward last layer have a diffent shape with label image");
	}
	int row = input.rows;
	int col = input.cols;

	Mat error(row, col, CV_32FC1);
#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			error.at<float>(i, j) = label.at<float>(i, j) - input.at<float>(i, j);
		}
	}

	out_error = error.clone();
	float sum = 0;
#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
#pragma omp critical
			{
				sum += error.at<float>(i, j);
			}
		}
	}

	sum /= (row * col);

	return sum;
}

float Loss::L2(Mat input, Mat label, Mat &out_error) {
	if (input.rows != label.rows || input.cols != label.cols) {
		fprintf(stderr, "Foward last layer have a diffent shape with label image");
	}
	int row = input.rows;
	int col = input.cols;
	Mat error(row, col, CV_32FC1);
#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			error.at<float>(i, j) = (label.at<float>(i, j) - input.at<float>(i, j)) * (label.at<float>(i, j) - input.at<float>(i, j));
		}
	}

	out_error = error.clone();
	float sum = 0;
#pragma omp parallel for num_threads(openmp_num_threads)
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
#pragma omp critical
			{
				sum += error.at<float>(i, j);
			}
		}
	}

	sum /= (row * col);

	return sum;
}