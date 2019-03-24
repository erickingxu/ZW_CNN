#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

namespace zwcnn {
	class Blob {
	public:
		Blob() {}
		string name;
		//生产这个blob的层的index
		int producer;
		//需要这个blob作为输入的层index
		vector <int> consumers;
	};
}
