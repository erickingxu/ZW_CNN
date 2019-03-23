#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Deconvolution {
private:
	int kernel_height;
	int kernel_width;
	int stride;
	string padding = "VALID";
	vector <Mat> input;
	vector <Mat> kernel;
public:
	Deconvolution(int kernel_height_, int kernel_width_, int stride_, string padding_, vector <Mat> input_, vector <Mat> kernel_) :
		kernel_height(kernel_height_), kernel_width(kernel_width_), stride(stride_), padding(padding_), input(input_), kernel(kernel_) {}
	vector <Mat> Activation();
};