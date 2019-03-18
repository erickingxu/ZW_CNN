#include "eltwise_layer.h"

vector <Mat> Eltwise::Activation() {
	if (input1.size() != input2.size()) {
		fprintf(stderr, "Eltwise Layer Dont Match!");
		exit(-1);
	}
	int Size = input1.size();
	vector <Mat> output;
	output.resize(Size);
	if (Type == "SUM") {
		for (int i = 0; i < Size; i++) {
			Mat op1 = input1[i];
			Mat op2 = input2[i];
			Mat dest(op1.rows, op1.cols, CV_32FC1);
			for (int j = 0; j < op1.rows; j++) {
				for (int k = 0; k < op1.cols; k++) {
					dest.at<float>(j, k) = op1.at<float>(j, k) + op2.at<float>(j, k);
				}
			}
			output[i] = dest;
		}
	}
	else if (Type == "MAX") {
		for (int i = 0; i < Size; i++) {
			Mat op1 = input1[i];
			Mat op2 = input2[i];
			Mat dest(op1.rows, op1.cols, CV_32FC1);
			for (int j = 0; j < op1.rows; j++) {
				for (int k = 0; k < op1.cols; k++) {
					dest.at<float>(j, k) = max(op1.at<float>(j, k), op2.at<float>(j, k));
				}
			}
			output[i] = dest;
		}
	}
	else if (Type == "MIN") {
		for (int i = 0; i < Size; i++) {
			Mat op1 = input1[i];
			Mat op2 = input2[i];
			Mat dest(op1.rows, op1.cols, CV_32FC1);
			for (int j = 0; j < op1.rows; j++) {
				for (int k = 0; k < op1.cols; k++) {
					dest.at<float>(j, k) = min(op1.at<float>(j, k), op2.at<float>(j, k));
				}
			}
			output[i] = dest;
		}
	}
	return output;
}