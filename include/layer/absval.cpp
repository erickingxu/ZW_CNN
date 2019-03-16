#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class AbsVal {
private:
	vector <Mat> input;
public:
	AbsVal(vector <Mat> input_) :input(input_) {}
	vector <Mat> Activation();
};
