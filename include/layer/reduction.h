#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Reduction {
private:
	vector <Mat> input;
	//caffe的dims默认是从1开始，这里保持一致
	int axis=1;
	string Type;
public:
	Reduction(vector<Mat>input_, int axis_, string Type_):input(input_), axis(axis_), Type(Type_){}
	vector <Mat> Activation();
};