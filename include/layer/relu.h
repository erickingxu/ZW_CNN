#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Relu {
private:
	vector<Mat> input;
public:
	Relu(vector<Mat> input_) :input(input_){}
	vector <Mat> Activation();
};
