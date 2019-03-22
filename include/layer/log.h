#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Log {
private:
	vector <Mat> input;
	float base;
	float scale;
	float shift;
public:
	Log(vector <Mat> input_, float base_=-1.0, float scale_=1.0, float shift_=0) :
		input(input_), base(base_), scale(scale_), shift(shift_){}
	vector <Mat> Activation();
};