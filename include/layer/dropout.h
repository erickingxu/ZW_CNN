#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

class Dropout {
private:
	vector <Mat> input;
public:
	Dropout(vector <Mat> input_):input(input_){}
	vector <Mat> Activation();
};