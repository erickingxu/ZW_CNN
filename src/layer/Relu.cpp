#include "Relu.h"
#include "config.h"

vector <Mat> Relu::Activation() {
	vector <Mat> output;
	int Size = input.size();
	for (int i = 0; i < Size; i++) {
		Mat temp(input[i].rows, input[i].cols, CV_32FC1);
		for (int j = 0; j < input[i].rows; j++) {
			for (int k = 0; k < input[i].cols; k++) {
				temp.at<float>(i, j) = input[i].at<float>(j, k) > 0 ? input[i].at<float>(j, k) : 0;
			}
		}
		output.push_back(temp);
	}
	return output;
}
