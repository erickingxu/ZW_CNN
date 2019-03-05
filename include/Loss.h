#pragma once
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Loss {
public:
	Loss() {};
	~Loss() {};
	float L1(Mat input, Mat label);
	float L2(Mat input, Mat label);
};