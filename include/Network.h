#pragma once
#include <iostream>
#include <ctime>
#include <random>
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
	//初始化策略
	//0表示[-1，1]随机初始化
	//1表示均值为GaussMean方差为GaussSigma的高斯初始化
	int InitType = 0;
	float GaussSigma = 1.0;
	float GaussMean = 0;
public:
	Net() {};
	~Net() {};
	void initNet(vector <int> layer_neuron_num_);
	float *random_uniform(int length);
	float *random_gaussian(int length);
	void initWeights();
	void initBiases();

};
