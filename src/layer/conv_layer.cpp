#include "conv_layer.h"
#include "padding.h"
#include "im2col_stride_1x1.h"
#include "im2col_stride_nxn.h"


vector<Mat> Convolution::Activation() {
	int input_length = input.size();
	if (padding == "SAME") {
		for (int i = 0; i < input_length; i++) {
			Padding ret(input[i], kernel_height, kernel_width, "SAME");
			input[i] = ret.Pad();
		}
	}
	if (kernel_stride_height == 1 && kernel_stride_width == 1) {
		int output_length = kernel.size();
		vector <Mat> output;
		output.resize(output_length);
		for (int i = 0; i < output_length; i += input_length) {
			if (input_length == 1) {
				IM2COL_1x1 ZW(input[i], kernel[i], kernel_stride_height, kernel_stride_width, padding);
				output[i] = ZW.ZW_IM2COL();
			}
			else {
				vector <Mat> temp;
				for (int j = 0; j < input_length; j++) {
					IM2COL_1x1 ZW(input[j], kernel[i + j], kernel_stride_height, kernel_stride_width, padding);
					temp.push_back(ZW.ZW_IM2COL());
				}
				for (int j = 1; j < input_length; j++) {
					temp[0] = temp[0] + temp[j];
				}
				output[i] = temp[0];
			}
		}
		return output;
	}
	else if (kernel_stride_height == kernel_stride_width && kernel_stride_height > 1) {
		int output_length = kernel.size();
		vector <Mat> output;
		for (int i = 0; i < output_length; i += input_length) {
			if (input_length == 1) {
				IM2COL_nxn ZW(input[i], kernel[i], kernel_stride_height, kernel_stride_width, padding);;
				output[i] = ZW.ZW_IM2COL();
			}
			else {
				vector <Mat> temp;
				for (int j = 0; j < input_length; j++) {
					IM2COL_nxn ZW(input[j], kernel[i + j], kernel_stride_height, kernel_stride_width, padding);
					temp.push_back(ZW.ZW_IM2COL());
				}
				for (int j = 1; j < input_length; j++) {
					temp[0] = temp[0] + temp[j];
				}
				output[i] = temp[0];
			}
		}
		return output;
	}
}
