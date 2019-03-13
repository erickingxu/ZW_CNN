#include "AbsVal.h"
#include "config.h"

Mat AbsVal::Activation(){
	int row = input.rows;
	int col = input.cols;
	Mat dest(row, col, CV_32FC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dest.at<float>(i, j) = input.at<float>(i, j) >= 0 ? input.at<float>(i, j) :  - (input.at<float>(i, j));
		}
	}

	return dest;

}