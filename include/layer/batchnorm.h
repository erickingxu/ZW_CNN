#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class BatchNorm{
private:
	vector <Mat> input;
	float running_mean;
	float running_var;
	float eps;
public:
	BatchNorm(vector <Mat> input_, float running_mean_, float running_var_, float eps_) :
		input(input_), running_mean(running_mean_), running_var(running_var_), eps(eps_) {}
	vector <Mat> Activation();
};