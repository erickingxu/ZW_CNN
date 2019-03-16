#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Sigmoid {
private:
	vector <Mat> input;
public:
	Sigmoid(vector <Mat> input_) :input(input_) {}
	vector <Mat> Activation();
};
