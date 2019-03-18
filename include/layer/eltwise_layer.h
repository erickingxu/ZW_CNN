#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class Eltwise {
private:
	vector <Mat> input1;
	vector <Mat> input2;
	string Type = "SUM";
public:
	Eltwise(vector<Mat>input1_, vector<Mat>input2_, string Type_):input1(input1_), input2(input2_), Type(Type_){}
	vector <Mat> Activation();
};