#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Concat {
private:
	vector<vector<Mat> > input;
public:
	Concat(vector<vector<Mat> >input_):input(input_){}
	vector <Mat> Activation();
};