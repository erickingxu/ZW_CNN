#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Convolution {
private:
	//num ������
	//n_in_channel ����ͨ������Ҳ��input.size()
	//n_out_channel ���ͨ������Ҳ��output.size()
	//padding pad�ķ�ʽ
	//kernel_width ����˵Ŀ�
	//kernel_height ����˵ĸ�
	//kernel_stride_height ��������еĻ�������
	//kernel_stride_width ��������еĻ�������
	//input ����
	//kenel �����
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
