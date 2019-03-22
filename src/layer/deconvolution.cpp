#include "deconvolution.h"

vector <Mat> Deconvolution::Activation() {
	int input_length = input.size();
	int output_length = kernel.size();
	vector <Mat> output;
	output.resize(output_length);
	for (int i = 0; i < output_length; i += input_length) {
		if (input_length == 1) {
			int output_height = (input[i].rows - 1) * kernel_height_stride + kernel_height;
			int output_width = (input[i].cols - 1) * kernel_width_stride + kernel_width;

		}
	}
}