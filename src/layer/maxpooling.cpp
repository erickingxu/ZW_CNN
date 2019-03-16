#include "maxpooling.h"


vector<Mat> MaxPooling::Pool() {
	if (kernel_height_stride != kernel_width_stride) {
		fprintf(stderr, "Pooling Stride Don't Match!");
		exit(-1);
	}
	int Size = input.size();
	vector <Mat> output;
	output.resize(Size);
	for (int i = 0; i < Size; i++) {
		int row = input[i].rows;
		int col = input[i].cols;
			Mat new_input(row / kernel_height_stride, col / kernel_width_stride, CV_32FC1);
			for (int j = 0; j < row; j += kernel_height_stride) {
				for (int k = 0; k < col; k += kernel_width_stride) {
					float maxx = -1e9;
					for (int x = 0; x < kernel_height; x++) {
						for (int y = 0; y < kernel_width; y++) {
							if ((i + x) >= 0 && (i + x) < row && (j + y) >= 0 && (j + y) < col) {
								maxx = max(maxx, input[i].at<float>(i + x, j + y));
							}
						}
					}
					new_input.at<float>(j / kernel_height_stride, k / kernel_width_stride) = maxx;
				}
			}
			output[i] = new_input;
	}
	return output;
}