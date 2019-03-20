#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Crop {
private:
	vector <Mat> input1;
	vector <Mat> input2;
	vector <int> crop_size;
	int axis = 1;
public:
	Crop(vector <Mat> input1_, vector <Mat> input2_, vector <int> crop_size_, int axis_):input1(input1_), input2(input2_),
		crop_size(crop_size_), axis(axis_){}
	vector <Mat> Activation();
};