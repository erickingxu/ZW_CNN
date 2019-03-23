#include "fractionally_padding.h"

vector <Mat> Fraction::Activation() {
	int Size = input.size();
	if (Size == 0) {
		fprintf(stderr, "Fractionally Input Is Empty!");
		exit(-1);
	}
	vector <Mat> output;
	output.resize(Size);
	for (int i = 0; i < Size; i++) {
		int row = input[i].rows;
		int col = input[i].cols;
		if (padding == "SAME") {
			int newrow = row + (kernel_height - 1) + (row - 1) * (stride - 1);
			int newcol = col + (kernel_width - 1) + (col - 1) * (stride - 1);
			Mat dest(newrow, newcol, CV_32FC1);
			for (int j = 0; j < newrow; j++) {
				for (int k = 0; k < newcol; k++) {
					if (j < (kernel_height - 1) / 2 || (newrow - j) <= (kernel_height - 1) / 2 || k < (kernel_width - 1) / 2 || (newcol - k) <= (kernel_width - 1) / 2) {
						dest.at<float>(j, k) = 0;
					}
					else if ((j - (kernel_height - 1) / 2) % stride == 0 && (k - (kernel_width - 1) / 2) % stride == 0) {
						dest.at<float>(j, k) = input[i].at<float>((j - (kernel_height - 1) / 2) / stride, (k - (kernel_width - 1) / 2) / stride);
					}
					else {
						dest.at<float>(j, k) = 0;
					}
				}
			}
			output[i] = dest;
		}
		else {
			int newrow = row + (row - 1) * (stride - 1);
			int newcol = col + (col - 1) * (stride - 1);
			Mat dest(newrow, newcol, CV_32FC1);
			for (int j = 0; j < newrow; j++) {
				for (int k = 0; k < newcol; k++) {
					if (j % stride == 0 && k % stride == 0) {
						dest.at<float>(j, k) = input[i].at<float>(j / stride, k / stride);
					}
					else {
						dest.at<float>(j, k) = 0;
					}
				}
			}
			output[i] = dest;
		}
	}
	return output;
}