#include "padding.h"

Mat Padding::Pad() {
	int row = input.rows;
	int col = input.cols;
	if (padding_type == "VALID") {
		return input;
	}
	else {
		int pad_x = (kernel_height - 1) >> 1;
		int pad_y = (kernel_width - 1) >> 1;
		int new_height = row + 2 * pad_x;
		int new_width = col + 2 * pad_y;
		Mat output(new_height, new_width, CV_32FC1);
		for (int i = 0; i < new_height; i++) {
			for (int j = 0; j < new_width; j++) {
				if (i <= pad_x || i + pad_x > new_height) {
					output.at<float>(i, j) = 0;
				}
				else {
					if (j <= pad_y || j + pad_y > new_width) {
						output.at<float>(i, j) = 0;
					}
					else {
						output.at<float>(i, j) = input.at<float>(i - pad_x, j - pad_y);
					}
				}
			}
		}
		return output;
	}
}