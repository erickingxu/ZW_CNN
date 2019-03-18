#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Split {
private:
	vector <Mat> input;
	int copy_size = 1;
public:
	Split(vector <Mat> input_, int copy_size_):input(input_), copy_size(copy_size_){}
	vector <vector<Mat>> Activation();
};