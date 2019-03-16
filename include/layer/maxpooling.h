#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class MaxPooling {
private:
	vector<Mat> input;
	int kernel_height;
	int kernel_width;
	int kernel_height_stride;
	int kernel_width_stride;
public:
	MaxPooling(vector<Mat> input_, int kernel_height_, int kernel_width_, int kernel_height_stride_, int kernel_width_stride_):
		input(input_), kernel_height(kernel_height_), kernel_width(kernel_width_), kernel_height_stride(kernel_height_stride_),
		kernel_width_stride(kernel_width_stride_){}
	vector <Mat> Pool();
};