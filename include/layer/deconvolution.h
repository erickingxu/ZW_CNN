#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Deconvolution {
private:
	int kernel_height;
	int kernel_width;
	int kernel_height_stride = 1;
	int kernel_width_stride = 1;
	vector <Mat> input;
	vector <Mat> kernel;
public:
	Deconvolution(int kernel_height_, int kernel_width_, int kernel_height_stride_, int kernel_width_stride_, vector <Mat> input_, vector <Mat> kernel_):
		kernel_height(kernel_height_), kernel_width(kernel_width_), kernel_height_stride(kernel_height_stride_), kernel_width_stride(kernel_width_stride_), input(input_), kernel(kernel_){}
	vector <Mat> Activation();
};