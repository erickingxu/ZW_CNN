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
			for (int i = 0; i < newrow; i++) {
				for (int j = 0; j < newcol; j++) {
					if (i < (kernel_height - 1) / 2 || (newrow - i) <= (kernel_height - 1) / 2 || j < (kernel_width - 1) / 2 || (newcol - j) <= (kernel_width - 1) / 2) {
						dest.at<float>(i, j) = 0;
					}
					else if ((i - (kernel_height - 1) / 2) % stride == 0 && (j - (kernel_width - 1) / 2) % stride == 0) {
						dest.at<float>(i, j) = input[i].at<float>((i - (kernel_height - 1) / 2) / stride, (j - (kernel_width - 1) / 2) / stride);
					}
					else {
						dest.at<float>(i, j) = 0;
					}
				}
			}
		}
		else {
			int newrow = row + (row - 1) * (stride - 1);
			int newcol = col + (col - 1) * (stride - 1);
			Mat dest(newrow, newcol, CV_32FC1);

		}
	}
}