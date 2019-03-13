#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Padding {
private:
	Mat input;
	int kernel_height;
	int kernel_width;
	string padding_type = "VALID";
public:
	Padding(Mat input_, int kernel_height_, int kernel_width_, string padding_type_):input(input_), kernel_height(kernel_height_), 
		kernel_width(kernel_width_), padding_type(padding_type_){}
	Padding(Mat input_, int kernel_height_, int kernel_width_) :input(input_), kernel_height(kernel_height_),
		kernel_width(kernel_width_) {}
	Mat Pad();
	
};