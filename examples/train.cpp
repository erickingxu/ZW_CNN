#include "Network.h"
#include "config.h"

int main() {
	vector <int> layer_neural_num = { 784, 200, 10 };
	Net net;
	net.InitType = 0;
	net.initNet(layer_neural_num);
	net.initWeights();
	net.initBiases();
	net.activation_func = "sigmoid";
	net.learning_rate = 0.3;
	net.loss_type = "L2";
	net.train_iter = 500;

	Mat input, label, test_input, test_label;
	//800个样本训练，200个样本测试
	int sample_num = 800;
	net.loadData("F:\\make_data\\input_1000_2.xml", input, label, 0, sample_num);
	net.loadData("F:\\make_data\\input_1000_2.xml", test_input, test_label, 800, 200);

	net.Train(input, label);
	net.Test(test_input, test_label);
	net.save_model("F:\\make_data\\sigmoid_800_200.xml");
	system("pause");

}