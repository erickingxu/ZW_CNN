#include "permute.h"

vector <Mat> Permute::Activation() {
	int Size = input.size();
	int row = input[0].rows;
	int col = input[0].cols;
	vector <Mat> output;
	
	if (order[0] == 0 && order[1] == 2){
		for (int c = 0; c < Size;++c){
			Mat dest(col, row, CV_32FC1);
			for (int w = 0; w < col; ++w){
				for (int h = 0; h < row; ++h){
					dest.at<float>(w, h) = input[c].at<float>(h, w);   
				}
			}
			output[c] = dest;
		}
	}
	else if (order[0] == 1 && order[1] == 0){
		for (int h = 0; h < row; ++h){
			Mat dest(Size, col, CV_32FC1);
			for (int c = 0; c < Size; ++c){
				for (int w = 0; w < col; ++w){
					dest.at<float>(c, w) = input[c].at<float>(h, w);
				}
			}
			output[h] = dest;
		}
	}
	else if (order[0] == 1 && order[1] == 2){
		for (int h = 0; h < row; ++h){
			Mat dest(col, Size, CV_32FC1);
			for (int w = 0; w < col; ++w){
				for (int c = 0; c < Size; ++c){
					dest.at<float>(w, c) = input[c].at<float>(h, w);
				}
			}
			output[h] = dest;
		}
	}
	else if (order[0] == 2 && order[1] == 0){
		for (int w = 0; w < col; ++w){
			Mat dest(Size, row, CV_32FC1);
			for (int c = 0; c < Size; ++c){
				for (int h = 0; h < row; ++h){
					dest.at<float>(c, h) = input[c].at<float>(h, w);
				}
			}
			output[w] = dest;
		}
	}
	else if (order[0] == 2 && order[1] == 1){
		for (int w = 0; w < col; ++w){
			Mat dest(row, Size, CV_32FC1);
			for (int h = 0; h < row; ++h){
				for (int c = 0; c < Size; ++c){
					dest.at<float>(h, c) = input[c].at<float>(h, w);
				}
			}
			output[w] = dest;
		}
	}
	else {
		for (int c = 0; c < Size; ++c){
			Mat dest(row, col, CV_32FC1);
			for (int h = 0; h < row; ++h){
				for (int w = 0; w < col; ++w){
					dest.at<float>(h, w) = input[c].at<float>(h, w);
				}
			}
			output[c] = dest;
		}
	}
	return output;
}