#include "im2col_stride_nxn.h"

Mat IM2COL_nxn::ZW_IM2COL() {
	int row = input.rows;
	int col = input.cols;
	int kernel_height = kernel.rows;
	int kernel_width = kernel.cols;
	//extend kernel
	Mat new_kernel(1, kernel_height * kernel_width, CV_32FC1);
	for (int i = 0; i < kernel_height; i++) {
		for (int j = 0; j < kernel_width; j++) {
			new_kernel.at<float>(0, i * kernel_height + j) = kernel.at<float>(i, j);
		}
	}
	//extend input
	if (padding_type == "VALID") {
		int st_x = (kernel_width - 1) / 2;
		int en_x = (row - st_x);
		int st_y = (kernel_height - 1) / 2;
		int en_y = (col - st_y);
		int index = 0;
		int new_row = ((row - kernel_height) / kernel_height_stride + 1) * ((col - kernel_width) / kernel_width_stride + 1);
		Mat new_input(new_row, kernel_height * kernel_width, CV_32FC1);
		for (int i = st_x; i < en_x; i += kernel_height_stride) {
			for (int j = st_y; j < en_y; j += kernel_width_stride) {
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
		Mat output((row - kernel_height) / kernel_height_stride + 1, (col - kernel_width) / kernel_width_stride + 1, CV_32FC1);
		for (int i = 0; i < (row - kernel_height) / kernel_height_stride + 1; i++) {
			for (int j = 0; j < (col - kernel_width) / kernel_width_stride + 1; j++) {
				output.at<float>(i, j) = ret.at<float>(0, i * ((col - kernel_width) / kernel_width_stride + 1) + j);
			}
		}
		return output;
	}
	else if (padding_type == "SAME") {
		int st_x = (kernel_height - 1) / 2;
		int en_x = row + (kernel_height - 1) / 2;
		int st_y = (kernel_width - 1) / 2;
		int en_y = col + (kernel_width - 1) / 2;
		int index = 0;
		int new_row = (row / kernel_height_stride) * (col / kernel_width_stride);
		Mat new_input(new_row, kernel_height * kernel_width, CV_32FC1);
		for (int i = st_x; i + kernel_height_stride < en_x; i += kernel_height_stride) {
			for (int j = en_x; j + kernel_width_stride < en_y; j += kernel_width_stride) {
				for (int x = 0; x < kernel_height; x++) {
					for (int y = 0; y < kernel_width; y++) {
						int oldi = i - (kernel_height - 1) / 2 + x;
						int oldj = j - (kernel_width - 1) / 2 + y;
						new_input.at<float>(index, x * kernel_width + y) = input.at<float>(oldi, oldj);
					}
				}
				index++;
			}
		}
		Mat ret = new_kernel * new_input;
		Mat output(row / kernel_height_stride, col / kernel_width_stride, CV_32FC1);
		for (int i = 0; i < row / kernel_height_stride; i++) {
			for (int j = 0; j < col / kernel_width_stride; j++) {
				output.at<float>(i, j) = ret.at<float>(0, i * (col / kernel_width_stride) + j);
			}
		}
		return output;
	}
}