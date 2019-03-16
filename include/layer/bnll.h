#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class BNLL {
private:
	vector <Mat> input;
public:
	BNLL(vector <Mat> input_) :input(input_) {}
	vector <Mat> Activation();
};
