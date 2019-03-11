#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Relu {
private:
	Mat input;
public:
	Relu(Mat in) :input(in){}
	Mat Activation();
	Mat DeActivation();
};