#include "conv_layer.h"

vector<vector<Mat>> Convolution::Activation() {
	vector <vector <Mat> > output;
	output.resize(num);
	//num个样本
	for (int i = 0; i < num; i++) {
		//单个样本
		vector <Mat> single_output;
		single_output.resize(n_out_channel);
		vector <Mat> single_input = input[i];
		vector <Mat> single_kernel = kernel[i];
		int Size = n_in_channel * n_out_channel;
		for (int j = 0; j < Size; j += n_in_channel) {
			vector <Mat> temp_kernel;
			temp_kernel.resize(n_in_channel);
			for (int k = 0; k < n_in_channel; k++) {
				temp_kernel[j + k] = single_kernel[j + k];
			}
		
		}
	}
}
