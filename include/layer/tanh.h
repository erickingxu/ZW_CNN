#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Tanh {
private:
	vector <Mat> input;
public:
	Tanh(vector <Mat> input_) :input(input_) {}
	vector <Mat> Activation();
};
