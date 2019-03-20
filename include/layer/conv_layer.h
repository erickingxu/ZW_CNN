#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Convolution {
private:
	//padding pad的方式
	//kernel_width 卷积核的宽
	//kernel_height 卷积核的高
	//kernel_stride_height 卷积核在行的滑动步长
	//kernel_stride_width 卷积核在列的滑动步长
	//input 输入
	//kenel 卷积核
	int kernel_height;
	int kernel_width;
	int kernel_stride_height = 1;
	int kernel_stride_width = 1;
	string padding = "VALID";
	vector <Mat>input;
	vector <Mat>kernel;
public:
	Convolution(vector <Mat>input_, vector <Mat>kernel_, int kernel_height_, int kernel_width_, int kernel_stride_height_, int kernel_stride_width_,
		string padding_) :input(input_), kernel(kernel_), kernel_height(kernel_height_), kernel_width(kernel_width),
		kernel_stride_height(kernel_stride_height_), kernel_stride_width(kernel_stride_width_), padding(padding_) {}
	Convolution(vector <Mat>input_, vector <Mat>kernel_, int kernel_height_, int kernel_width_) :input(input_), kernel(kernel_), kernel_height(kernel_height_), kernel_width(kernel_width) {}
	vector<Mat> Activation();
};