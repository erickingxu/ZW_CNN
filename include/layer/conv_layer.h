#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Convolution {
private:
	//num 样本数
	//n_in_channel 输入通道数，也是input.size()
	//n_out_channel 输出通道数，也是output.size()
	//padding pad的方式
	//kernel_width 卷积核的宽
	//kernel_height 卷积核的高
	//kernel_stride_height 卷积核在行的滑动步长
	//kernel_stride_width 卷积核在列的滑动步长
	//input 输入
	//kenel 卷积核
	int num;
	int n_in_channel;
	int n_out_channel;
	int kernel_height;
	int kernel_width;
	int kernel_stride_height = 1;
	int kernel_stride_width = 1;
	bool padding = true;
	vector <vector <Mat> >input;
	vector <vector <Mat> >kernel;
public:
	Convolution(vector <vector <Mat> >input_, vector <vector <Mat> >kernel_, int n_in_channel_, int n_out_channel_, int kernel_height_, int kernel_width_, int kernel_stride_height_, int kernel_stride_width_,
		bool padding_):input(input_), kernel(kernel_), n_in_channel(n_in_channel_), n_out_channel(n_out_channel_), kernel_height(kernel_height_), kernel_width(kernel_width), 
		kernel_stride_height(kernel_stride_height_), kernel_stride_width(kernel_stride_width_), padding(padding_){}
	Convolution(vector <vector <Mat> >input_, vector <vector <Mat> >kernel_, int n_in_channel_, int n_out_channel_, int kernel_height_, int kernel_width_) :input(input_), kernel(kernel_), n_in_channel(n_in_channel_),
		n_out_channel(n_out_channel_), kernel_height(kernel_height_), kernel_width(kernel_width){}
	vector<vector<Mat> > Activation();
};
