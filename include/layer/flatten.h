#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Flatten {
private:
	vector <Mat> input;
public:
	Flatten(vector <Mat> input_):input(input_){}
	Mat Activation();
};