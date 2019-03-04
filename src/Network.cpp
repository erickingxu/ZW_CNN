#include "../include/Network.h"

void Net::initNet(vector <int> layer_neuron_num_) {
	//赋值
	layer_neuron_num = layer_neuron_num_;
	//构造layer，由于隐藏层只需要神经元个数，第二维默认为1，并选择初始化方式
	int Size = layer_neuron_num.size();
	layer.resize(Size);
	for (int i = 0; i < Size; i++) {
		Mat temp(layer_neuron_num[i], 1, CV_32FC1);
		layer.emplace_back(temp);
	}
	//构造weights和biases
	weights.resize(Size - 1);
	biases.resize(Size - 1);
	for (int i = 0; i < Size; i++) {
		Mat temp_weight(layer[i + 1].rows, layer[i].rows, CV_32FC1);
		Mat temp_biases(layer[i + 1].rows, layer[i].rows, CV_32FC1);
		weights.emplace_back(temp_weight);
		biases.emplace_back(temp_biases);
	}
	cout << "============= Initialize zxy_neural_network Net Done! ====================" << endl;
}

float *Net::random_uniform(int length) {
	float *dest = new float[length];
	uniform_real_distribution<double> dist(-1.0, 1.0);
	mt19937 rng;
	rng.seed(random_device{}());
	for (int i = 0; i < length; i++) {
		dest[i] = dist(rng);
	}
	return dest;
}

inline float uniform_rand(float l, float r) {
	return l + ((double)std::rand() / (RAND_MAX + 1.0)) * (r - l);
}

float *Net::random_gaussian(int length) {
	float *dest = new float[length];
	for (int i = 0; i < length; i++) {
		double x, y, r2;
		do {
			x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
			y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
			r2 = x * x + y * y;
		} while (r2 > 1.0 || r2 == 0.0);
		dest[i] = Net::GaussMean + Net::GaussSigma * y * std::sqrt(-2.0 * log(r2) / r2);
	}
	return dest;
}

void Net::initWeights() {
	int Size = weights.size();
	for (int i = 0; i < Size; i++) {
		int row = weights[i].rows;
		int col = weights[i].cols;
		float *data = new float[row * col];
		if (Net::InitType == 0) {
			data = Net::random_uniform(row * col);
		}
		else if (Net::InitType == 1) {
			data = Net::random_gaussian(row * col);
		}
		Mat temp(row, col, CV_32FC1, data);
		weights[i] = temp.clone();
	}
	cout << "============= Initialize zxy_neural_network Weights Done! ====================" << endl;
}

void Net::initBiases() {
	int Size = biases.size();
	for (int i = 0; i < Size; i++) {
		int row = biases[i].rows;
		int col = biases[i].cols;
		float *data = new float[row * col];
		if (Net::InitType == 0) {
			data = Net::random_uniform(row * col);
		}
		else if (Net::InitType == 1) {
			data = Net::random_gaussian(row * col);
		}
		Mat temp(row, col, CV_32FC1, data);
		biases[i] = temp.clone();
	}
	cout << "============= Initialize zxy_neural_network Biases Done! ====================" << endl;
}

