#include "Network.h"

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
	cout << "============= Initialize zxy_neural_network Done! ====================" << endl;
}

