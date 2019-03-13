#pragma
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class BNLL {
private:
	Mat input;
public:
	BNLL(Mat in) :input(in) {}
	Mat Activation();
	Mat DeActivation();
};