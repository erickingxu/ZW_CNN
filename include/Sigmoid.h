#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Sigmoid {
private:
	Mat input;
public:
	Sigmoid(Mat in) :input(in) {}
	Mat Activation();
	Mat DeActivation();
};