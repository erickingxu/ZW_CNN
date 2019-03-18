#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Slice {
private:
	vector <Mat> input;
	vector <int> slice_point;
public:
	Slice(vector <Mat> input_, vector <int> slice_point_):input(input_), slice_point(slice_point_){}
	vector <vector <Mat> > Activation();
};