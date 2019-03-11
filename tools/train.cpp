#include "Network.h"

int main() {
	vector <int> layer_neural_num = { 784, 200, 10 };
	Net net;
	net.InitType = 0;
	net.initNet(layer_neural_num);
	net.initWeights();
	net.initBiases();
	net.activation_func = "sigmoid";
	net.learning_rate = 0.03;
	net.loss_type = "L2";
	net.train_iter = 2000;

	Mat input, label, test_input, test_label;
	//800个样本训练，200个样本测试
	int sample_num = 800;
	net.loadData("C:\\Users\\xiaoyu\\Desktop\\zxy_neural_network\\data\\input_1000.xml", input, label, 0, sample_num);
	net.loadData("C:\\Users\\xiaoyu\\Desktop\\zxy_neural_network\\data\\input_1000.xml", test_input, test_label, 800, 200);

	net.Train(input, label);
	net.Test(test_input, test_label);
	net.save_model("C:\\Users\\xiaoyu\\Desktop\\zxy_neural_network\\models\\sigmoid_800_200.xml");
	system("pause");
	return 0;
}