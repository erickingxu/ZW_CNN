#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Reshape {
private:
	vector <Mat> input;
	int dim_c;
	int dim_h;
	int dim_w;
public:
	Reshape(vector <Mat> input_, int dim_c_, int dim_h_, int dim_w_) :
		    input(input_), dim_c(dim_c_), dim_h(dim_h_), dim_w(dim_w_) {}
	vector <Mat> Activation();
};