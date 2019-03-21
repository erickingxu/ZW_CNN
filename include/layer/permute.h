#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Permute {
private:
	vector <Mat> input;
	vector <int> order;
public:
	Permute(vector <Mat> input_,  vector <int> order_) :input(input_), order(order_){}
	vector <Mat> Activation();
};