#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Scale{
private:
	vector <Mat> input;
	float gamma;
	float beta;
public:
	Scale(vector <Mat> input_, float gamma_, float beta_) :
		input(input_), gamma(gamma_), beta(beta_) {}
	vector <Mat> Activation();
};