#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Inner_Product {
private:
	Mat input;
	Mat weights;
public:
	Inner_Product(Mat input_, Mat weights_):input(input_), weights(weights_){}
	Mat Activation();
};