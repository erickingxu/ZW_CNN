#include "im2col_stride_1x1.h"

Mat IM2COL_1x1::ZW_IM2COL() {
	int row = input.rows;
	int col = input.cols;
	int kernel_height = kernel.rows;
	int kernel_width = kernel.cols;
	//??kernel
	Mat new_kernel(1, kernel_height * kernel_width, CV_32FC1);
	for (int i = 0; i < kernel_height; i++) {
		for (int j = 0; j < kernel_width; j++) {
			new_kernel.at<float>(0, i * kernel_height + j) = kernel.at<float>(i, j);
		}
	}
	//??input
	if (padding_type == "VALID") {
		int st_x = (kernel_width - 1) / 2;
		int en_x = (row - st_x);
		int st_y = (kernel_height - 1) / 2;
		int en_y = (col - st_y);
		int index = 0;
		int new_row = (row - kernel_height) * (col - kernel_width);
		Mat new_input(new_row, kernel_height * kernel_width, CV_32FC1);
		for (int i = st_x; i < en_x; i++) {
			for (int j = en_x; j < en_y; j++) {
				for (int x = 0; x < kernel_height; x++) {
					for (int y = 0; y < kernel_width; y++) {
						int oldi = i - st_x + x;
						int oldj = j - st_y + y;
						new_input.at<float>(index, x * kernel_width + y) = input.at<float>(oldi, oldj);
					}
				}
				index++;
			}
		}
		Mat ret = new_kernel * new_input;
		Mat output(row, col, CV_32FC1);
		for (int i = 0; i < (row - kernel_height); i++) {
			for (int j = 0; j < (col - kernel_width); j++) {
				output.at<float>(i, j) = ret.at<float>(0, i * col + j);
			}
		}
		return output;
	}
	else if (padding_type == "SAME") {
		int st_x = 0;
		int en_x = row;
		int st_y = 0;
		int en_y = col;
		int index = 0;
		int new_row = row * col;
		Mat new_input(new_row, kernel_height * kernel_width, CV_32FC1);
		for (int i = st_x; i < en_x; i++) {
			for (int j = en_x; j < en_y; j++) {
				for (int x = 0; x < kernel_height; x++) {
					for (int y = 0; y < kernel_width; y++) {
						int oldi = i - (kernel_height - 1) / 2 + x;
						int oldj = j - (kernel_width - 1) / 2 + y;
						if (oldi < 0 || oldj < 0 || oldi >= row || oldj >= col) {
							new_input.at<float>(index, x * kernel_width + y) = 0;
						}
						else {
							new_input.at<float>(index, x * kernel_width + y) = input.at<float>(oldi, oldj);
						}
					}
				}
				index++;
			}
		}
		Mat ret = new_kernel * new_input;
		Mat output(row , col, CV_32FC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				output.at<float>(i, j) = ret.at<float>(0, i * col + j);
			}
		}
		return output;
	}
}