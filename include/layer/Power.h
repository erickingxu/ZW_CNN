#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Power {
private:
	Mat input;
	float shift;
	float scale;
	float power;
public:
	~Power() {};
	Power(Mat in, float sht=0, float scl=1.0, float pow=1.0) : input(in), shift(sht), scale(scl), power(pow){}
	Mat Activation();
	Mat DeActivation();
};