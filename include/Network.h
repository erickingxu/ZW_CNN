#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

class Net
{
public:
	//每一层的神经元数目
	vector <int> layer_neuron_num;
	//层，使用Opencv的Mat
	vector <Mat> layer;
	//权值矩阵
	vector <Mat> weights;
	//偏置矩阵
	vector <Mat> biases;
public:
	Net() {};
	~Net() {};
	void initNet(vector <int> layer_neuron_num_);

};
