#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Power {
private:
	Mat input;

public:
	Power(Mat in) : input(in) {}
	Mat Activation(float shift=0, float scale=1.0, float power=1.0);
	Mat DeActivation(float shift = 0, float scale = 1.0, float power = 1.0);
};