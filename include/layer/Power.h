#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Power {
private:
	vector <Mat> input;
	float shift;
	float scale;
	float power;
public:
	Power(vector <Mat> input_, float shift_ = 0, float scale_ = 1.0, float power_ = 1.0) : 
		input(input_), shift(shift_), scale(scale_), power(power_) {}
	vector <Mat> Activation();
};