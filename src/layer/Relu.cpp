#include "Relu.h"
#include "config.h"

vector <Mat> Relu::forward() {
	vector <Mat> output;
	output.resize(input.size());
	for (int i = 0; i < input.size(); i++) {
		int row = input[i].rows;
		int col = input[i].cols;
		Mat dest(row, col, CV_32FC1);
#pragma omp parallel for num_threads(openmp_num_threads)
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				dest.at<float>(j, k) = input[i].at<float>(j, k) > 0 ? input[i].at<float>(j, k) : 0;
			}
		}
		output[i] = dest;
	}
	return output;
}

