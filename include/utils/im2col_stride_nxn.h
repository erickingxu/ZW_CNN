#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class IM2COL_nxn {
private:
	Mat input;
	Mat kernel;
	int kernel_height_stride;
	int kernel_width_stride;
	string padding_type = "VALID";
public:
	IM2COL_nxn(Mat input_, Mat kernel_, int kernel_height_stride_, int kernel_width_stride_, string padding_type_) :input(input_), kernel(kernel_),
		kernel_height_stride(kernel_height_stride_), kernel_width_stride(kernel_width_stride_), padding_type(padding_type_) {}
	Mat ZW_IM2COL();
};