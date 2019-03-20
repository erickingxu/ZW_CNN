#include "crop.h"

vector <Mat> Crop::Activation() {
	if (axis == 1) {
		if (crop_size.size() < 3) {
			fprintf(stderr, "Input Don't Match!");
			exit(-1);
		}
		int c = crop_size[0];
		int h = crop_size[1];
		int w = crop_size[2];
		vector <Mat> output;
		output.resize(input2.size());
		for (int i = c; i < c + input2.size(); i++) {
			Mat dest(input2[0].rows, input2[0].cols, CV_32FC1);
			for (int j = h; j < h + input2[0].rows; j++) {
				for (int k = w; k < w + input2[0].cols; k++) {
					dest.at<float>(j - h, k - w) = input1[i].at<float>(j, k);
				}
			}
			output[i - c] = dest;
		}
		return output;
	}
	else if (axis == 2) {
		if (crop_size.size() < 2) {
			fprintf(stderr, "Input Don't Match!");
			exit(-1);
		}
		int h = crop_size[0];
		int w = crop_size[1];
		vector <Mat> output;
		output.resize(input1.size());
		for (int i = 0; i < input1.size(); i++) {
			Mat dest(input2[0].rows, input2[0].cols, CV_32FC1);
			for (int j = h; j < h + input2[0].rows; j++) {
				for (int k = w; k < w + input2[0].cols; k++) {
					dest.at<float>(j - h, k - w) = input1[i].at<float>(j, k);
				}
			}
			output[i] = dest;
		}
		return output;
	}
	else if (axis == 3) {
		if (crop_size.size() < 1) {
			fprintf(stderr, "Input Don't Match!");
			exit(-1);
		}
		int w = crop_size[0];
		vector <Mat> output;
		output.resize(input1.size());
		for (int i = 0; i < input1.size(); i++) {
			Mat dest(input1[0].rows, input2[0].cols, CV_32FC1);
			for (int j = 0; j < input1[0].rows; j++) {
				for (int k = w; k < w + input2[0].cols; k++) {
					dest.at<float>(j, k - w) = input1[i].at<float>(j, k);
				}
			}
			output[i] = dest;
		}
		return output;
	}
}