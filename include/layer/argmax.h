#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class ArgMax {
private:
	Mat input;
public:
	ArgMax(Mat input_):input(input_){}
	int Activation();
};