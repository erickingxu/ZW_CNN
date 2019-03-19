#include "reduction.h"

vector <Mat> Reduction::Activation() {
	if (input.size() == 0) {
		fprintf(stderr, "Input Is Empty");
		exit(-1);
	}
	int Size = input.size();
	if (axis == 1) {
		return input;
	}
	else if (axis == 2) {
		vector <Mat> output;
		output.resize(1);
		Mat dest(1, 1, CV_32FC1);
		if (Type == "SUM") {
			dest.at<float>(0, 0) = 0;
		}
		else if (Type == "MEAN") {
			dest.at<float>(0, 0) = 0;
		}
		for (int i = 0; i < Size; i++) {
			for (int j = 0; j < input[i].rows; j++) {
				for (int k = 0; k < input[i].cols; k++) {
					if (Type == "SUM") {
						dest.at<float>(0, 0) += input[i].at<float>(j, k);
					}
					else if (Type == "MEAN") {
						dest.at<float>(0, 0) += input[i].at<float>(j, k);
					}
				}
			}
		}
		if (Type == "MEAN") {
			dest.at<float>(0, 0) = dest.at<float>(0, 0) / input[0].rows / input[0].cols / Size;
		}
		output[0] = dest;
		return output;
	}
	else if (axis == 3) {
		vector <Mat> output;
		output.resize(Size);
		for (int i = 0; i < Size; i++) {
			Mat dest(1, 1, CV_32FC1);
			if (Type == "SUM") {
				dest.at<float>(0, 0) = 0;
			}
			else if (Type == "MEAN") {
				dest.at<float>(0, 0) = 0;
			}
			for (int j = 0; j < input[i].rows; j++) {
				for (int k = 0; k < input[i].cols; k++) {
					if (Type == "SUM") {
						dest.at<float>(0, 0) += input[i].at<float>(j, k);
					}
					else if (Type == "MEAN") {
						dest.at<float>(0, 0) += input[i].at<float>(j, k);
					}
				}
			}
			output[i] = dest;
		}
		return output;
	}
	else if (axis == 4) {
		vector <Mat> output;
		output.resize(Size);
		for (int i = 0; i < Size; i++) {
			Mat dest(1, input[i].cols, CV_32FC1);
			for (int j = 0; j < input[i].cols; j++) {
				dest.at<float>(0, j) = 0;
			}
			for (int j = 0; j < input[i].rows; j++) {
				for (int k = 0; k < input[i].cols; k++) {
					if (Type == "SUM") {
						dest.at<float>(0, k) += input[i].at<float>(j, k);
					}
					else if (Type == "MEAN") {
						dest.at<float>(0, k) += input[i].at<float>(j, k);
					}
				}
			}
			output[i] = dest;
		}
		return output;
	}
}