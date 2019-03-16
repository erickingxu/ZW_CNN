#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Inner_Product {
private:
	vector <Mat> input;
	vector <Mat> weights;
public:
	Inner_Product(vector <Mat> input_, vector <Mat> weights_):inputs(input_), weights(weights_){}
	vector <Mat> Activation();
};