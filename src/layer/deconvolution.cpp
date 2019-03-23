#include "deconvolution.h"
#include "fractionally_padding.h"
#include "im2col_stride_1x1.h"
#include "im2col_stride_nxn.h"

vector <Mat> Deconvolution::Activation() {
	int input_length = input.size();
	int output_length = kernel.size();
	vector <Mat> output;
	output.resize(output_length);
	for (int i = 0; i < output_length; i += input_length) {
		vector <Mat> temp;
		temp.resize(input_length);
		for (int j = i; j < i + input_length; j++) {
			temp[j] = input[j];
		}
		Fraction frac(temp, stride, padding, kernel_height, kernel_width);
		vector <Mat> out = frac.Activation();
		if (input_length == 1) {
			IM2COL_1x1 ZW(out[0], kernel[i], stride, stride, padding);
			output[i] = ZW.ZW_IM2COL();
		}
		else {
			vector <Mat> out;
			for (int j = 0; j < input_length; j++) {
				IM2COL_1x1 ZW(input[j], kernel[i + j], stride, stride, padding);
				out.push_back(ZW.ZW_IM2COL());
			}
			for (int j = 1; j < input_length; j++) {
				out[0] = out[0] + out[j];
			}
			output[i] = out[0];
		}
	}
	return output;
}