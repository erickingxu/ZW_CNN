#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Fraction {
private:
	vector <Mat> input;
	int stride;
	string padding = "VALID";
	int kernel_height;
	int kernel_width;
public:
	Fraction(vector <Mat> input_, int stride_, string padding_, int kernel_height_, int kernel_width_):input(input_), stride(stride_), 
		padding(padding_), kernel_height(kernel_height_), kernel_width(kernel_width_){}
	vector <Mat> Activation();
};