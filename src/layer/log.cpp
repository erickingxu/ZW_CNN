#include "log.h"

vector <Mat> Log::Activation() {
	int Size = input.size();
	int row = input[0].rows;
	int col = input[0].cols;

	vector <Mat> output;
	output.resize(Size);
	
	if (base == -1.0){
		for (int i = 0; i < Size; i++) {
			Mat dest(row, col, CV_32FC1);
			for (int j = 0; j < row; j++) {
				for (int k = 0; k < col; k++) {
					dest.at<float>(j, k) = log(shift + scale * input[i].at<float>(j, k));
	            }
			}
			output[i] = dest;
		}
	}
	else if (base > 0){
		for (int i = 0; i < Size; i++) {
			Mat dest(row, col, CV_32FC1);
			for (int j = 0; j < row; j++) {
				for (int k = 0; k < col; k++) {
					dest.at<float>(j, k) = log(shift + scale * input[i].at<float>(j, k)) / log(base);
				}
			}
			output[i] = dest;
		}
	}
	else{
		fprintf(stderr, "Input parameter error!");
		exit(-1);
	}
	return output;
}