#include "reshape.h"

vector <Mat> Reshape::Activation() {
	if (input.size() == 0) {
		fprintf(stderr, "Input Is Empty!");
		exit(-1);
	}
	if ((dim_c == -1 && dim_h == -1) || (dim_c == -1 && dim_w == -1) || (dim_h == -1 && dim_w == -1)){
		fprintf(stderr, "Input Parameter Error!");
		exit(-1);
	}
	 
	int Size = input.size();
	int row = input[0].rows;
	int col = input[0].cols;
	Mat src(Size * row * col, 1, CV_32FC1);
	int index = 0;
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				src.at<float>(index, 0) = input[i].at<float>(i, j);
				index++;
			}
		}
	}
	
	index = 0;
	int Out_Size;
	int Out_row;
	int Out_col;
	vector <Mat> output;
	if (dim_c == 0) 
		Out_Size = Size;
	else
		Out_Size = dim_c;

	if (dim_h == 0)
		Out_row = row;
	else
		Out_row = dim_h;

	if (dim_w == 0)
		Out_col = col;
	else
		Out_col = dim_w;

	 if (dim_c == -1)
		 Out_Size = Size*row*col / Out_row / Out_col;

	 if (dim_h == -1)
		 Out_row = Size*row*col / Out_Size / Out_col;

	 if (dim_w == -1)
		 Out_col = Size*row*col / Out_Size / Out_row;

	for (int i = 0; i < Out_Size; i++) {
		Mat dest(Out_row, Out_col, CV_32FC1);
		for (int j = 0; j < Out_row; j++) {
			for (int k = 0; k < Out_col; k++) {
				dest.at<float>(j, k) = src.at<float>(index, 0);
				index++;
			}
		}
		output[i] = dest;
	}
	return output;
}