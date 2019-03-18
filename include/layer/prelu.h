#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class PRelu {
private:
	vector <Mat> input;
	float alpha;
public:
	PRelu(vector <Mat> input_, float alpha_):input(input_), alpha(alpha_){}
	vector <Mat> Activation();
};