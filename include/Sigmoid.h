#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Sigmoid {
public:
	Sigmoid();
	~Sigmoid();
	Mat Activation(Mat input) {
		int row = input.rows;
		int col = input.cols;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {

			}
		}
	}

};