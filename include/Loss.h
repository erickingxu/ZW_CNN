#pragma once
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Loss {
private:
	Mat input_;
	Mat label_;
public:
	Loss(Mat in, Mat la) :input_(in), label_(la) {}
	float L1(Mat &out_error);
	float L2(Mat &out_error);
};
