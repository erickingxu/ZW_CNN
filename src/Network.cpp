#include "Network.h"
#include "Sigmoid.h"
#include "Tanh.h"
#include "Relu.h"
#include "Loss.h"

void Net::initNet(vector <int> layer_neuron_num_) {
	//赋值
	layer_neuron_num = layer_neuron_num_;
	//构造layer，由于隐藏层只需要神经元个数，第二维默认为1，并选择初始化方式
	int Size = layer_neuron_num.size();
	for (int i = 0; i < Size; i++) {
		Mat temp(layer_neuron_num[i], 1, CV_32FC1);
		layer.push_back(temp);
	}
	//构造weights和biases
	for (int i = 0; i < Size; i++) {
		Mat temp_weight(layer[i + 1].rows, layer[i].rows, CV_32FC1);
		Mat temp_biases(layer[i + 1].rows, layer[i].rows, CV_32FC1);
		weights.push_back(temp_weight);
		biases.push_back(temp_biases);
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

void Net::SetThreads(int num) {
	openmp_num_threads = num;
}

void Net::SetActivation(string input) {
	activation_func = input;
}

void Net::forward() {
	int Size = layer_neuron_num.size();

	//线性模型Y = f(XW + b)
	for (int i = 0; i < Size - 1; i++) {
		Mat output = weights[i] * layer[i] + biases[i];
		//非线性函数
		if (activation_func == "sigmoid") {
			Sigmoid sigmoid;
			layer[i + 1] = sigmoid.Activation(output);
		}
		else if (activation_func == "relu") {
			Relu relu;
			layer[i + 1] = relu.Activation(output);
		}
		else if (activation_func == "tanh") {
			Tanh tanh;
			layer[i + 1] = tanh.Activation(output);
		}
	}
	Loss now;
	loss = now.L1(layer[Size - 1], label, output_error);
}

// 梯度矩阵的shape和网络层mat的shape完全一样
void Net::getGrad() {
	int Size = layer.size() - 1;
	delta.resize(Size);
	for (int i = Size; i >= 0; i--) {
		delta[i].create(layer[i + 1].size(), CV_32FC1);
		Mat dx;
		if (activation_func == "sigmoid") {
			Sigmoid sigmoid;
			dx = sigmoid.DeActivation(layer[i + 1]);
		}
		else if (activation_func == "relu") {
			Relu relu;
			dx = relu.DeActivation(layer[i + 1]);
		}
		else if (activation_func == "tanh") {
			Tanh tanh;
			dx = tanh.DeActivation(layer[i + 1]);
		}
		if (i == Size) {
			delta[i] = dx.mul(output_error);
		}
		else {
			//链式法则
			delta[i] = dx.mul(weights[i + 1].t() * delta[i + 1]);
		}
	}
}

void Net::UpdateParameters() {
	int Size = weights.size();
	for (int i = 0; i < Size; i++) {
		Mat delta_weights = learning_rate * (delta[i] * layer[i].t());
		Mat delta_biases = learning_rate * delta[i];
		weights[i] = weights[i] + delta_weights;
		biases[i] = biases[i] + delta_biases;
	}
}

void Net::backward() {
	getGrad();
	UpdateParameters();
}






