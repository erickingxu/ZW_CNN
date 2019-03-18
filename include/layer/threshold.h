#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Threshold {
private:
	vector <Mat> input;
	float threshold;
public:
	Threshold(vector <Mat> input_, float threshold_):input(input_), threshold(threshold_){}
	vector <Mat> Activation();
};