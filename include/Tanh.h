#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Tanh {
private:
	Mat input;
public:
	Tanh(Mat in) :input(in) {}
	Mat Activation();
	Mat DeActivation();
};