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
	layer.clear();
	for (int i = 0; i < Size; i++) {
		Mat temp(layer_neuron_num[i], 1, CV_32FC1);
		layer.push_back(temp);
	}
	//构造weights和biases
	weights.clear();
	biases.clear();
	for (int i = 0; i < Size - 1; i++) {
		Mat temp_weight(layer[i + 1].rows, layer[i].rows, CV_32FC1);
		Mat temp_biases(layer[i + 1].rows, 1, CV_32FC1);
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
		if (InitType == 0) {
		data = random_uniform(row * col);
		}
		else if (InitType == 1) {
		data = random_gaussian(row * col);
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
		if (InitType == 0) {
		data = random_uniform(row * col);
		}
		else if (InitType == 1) {
		data = random_gaussian(row * col);
		}
		Mat temp(row, col, CV_32FC1, data);
		biases[i] = temp.clone();
	}
	cout << "============= Initialize zxy_neural_network Biases Done! ====================" << endl;
}

void Net::SetActivation(string input) {
	activation_func = input;
}

void Net::forward() {
	int Size = layer_neuron_num.size();

	//线性模型Y = f(XW + b)
	for (int i = 0; i < Size - 1; i++) {
		//printf("%d %d\n", weights[i].rows, weights[i].cols);
		//printf("%d %d\n", layer[i].rows, layer[i].cols);
		//printf("%d %d\n", biases[i].rows, biases[i].cols);

		Mat output = weights[i] * layer[i] + biases[i];

		//非线性函数
		if (activation_func == "sigmoid") {
			Sigmoid sigmoid(output);
			layer[i + 1] = sigmoid.Activation();
		}
		else if (activation_func == "relu") {
			Relu relu(output);
			layer[i + 1] = relu.Activation();
		}
		else if (activation_func == "tanh") {
			Tanh tanh(output);
			layer[i + 1] = tanh.Activation();
		}
	}

	Loss now(layer[Size - 1], label);
	loss = now.L2(output_error);
}

// 梯度矩阵的shape和网络层mat的shape完全一样
void Net::getGrad() {
	int Size = layer.size() - 1;
	delta.resize(Size);
	for (int i = Size - 1; i >= 0; i--) {
		delta[i].create(layer[i + 1].size(), CV_32FC1);
		Mat dx(layer[i + 1].rows, layer[i + 1].cols, CV_32FC1);
		if (activation_func == "sigmoid") {
			Sigmoid sigmoid(layer[i + 1]);
			dx = sigmoid.DeActivation();
		}
		else if (activation_func == "relu") {
			Relu relu(layer[i + 1]);
			dx = relu.DeActivation();
		}
		else if (activation_func == "tanh") {
			Tanh tanh(layer[i + 1]);
			dx = tanh.DeActivation();
		}

		if (i == Size - 1) {
			delta[i] = dx.mul(output_error);
		}
		else {
			//链式法则
			delta[i] = dx.mul((weights[i + 1].t()) * delta[i + 1]);
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

void Net::Train(Mat input, Mat label_) {
	if (input.rows == 0 || input.cols == 0) {
		fprintf(stderr, "Input Is Empty!");
		return;
	}
	if (input.rows != layer[0].rows) {
		fprintf(stderr, "Rows of input don't Match the number of input!");
		return;
	}
	cout << "*********Train zxy_neural_network Begin!***********" << endl;
	int row = input.rows;
	int col = input.cols;
	//单个样本即是batch为1
	if (col == 1) {
		label = label_;
		layer[0] = input;
		for (int i = 1; i <= train_iter; i++) {
			forward();
			backward();
			if (i % 1 == 0 || i == train_iter) {
				cout << "Train " << i << " times" << endl;
				cout << "Loss: " << loss << endl;
			}
		}
		cout << endl << "Train " << train_iter << " times" << endl;
		cout << "Finally Loss: " << loss << endl;
		cout << "Train sucessfully!" << endl;
	}
	else {
		float final_loss = 0;
		for (int i = 1; i <= train_iter; i++) {
			float batch_loss = 0;
			for (int j = 0; j < col; j++) {
				label = label_.col(j);
				layer[0] = input.col(j);

				/*for (int jj = 0; jj < layer[0].rows; jj++) {
				for (int kk = 0; kk < layer[0].cols; kk++) {
				printf("%.5f ", layer[0].at<float>(jj, kk));
				}
				printf("\n");
				}*/

				forward();

				batch_loss += loss;
				backward();
			}
			//if (i % 1 == 0 || i == train_iter) {
			cout << "Train " << i << " times" << endl;
			cout << "Loss: " << batch_loss << endl;
			//}
			final_loss = batch_loss;
		}
		final_loss /= train_iter;
		cout << endl << "Train " << train_iter << " times" << endl;
		cout << "Finally Loss: " << final_loss << endl;
		cout << "Train sucessfully!" << endl;
	}
}

void Net::Test(Mat input, Mat label_) {
	if (input.rows == 0 || input.cols == 0) {
		fprintf(stderr, "Input Is Empty!");
		return;
	}
	cout << "*********Test zxy_neural_network Begin!***********" << endl;
	if (input.rows != layer[0].rows) {
		fprintf(stderr, "Rows of input don't Match the number of input!");
		return;
	}
	int row = input.rows;
	int col = input.cols;
	int label_row = label_.rows;
	int label_col = label_.cols;
	//对一张图片进行预测
	if (col == 1) {
		int predict_index = Predict1(input);

		int label_index = 0;
		float mx_score = 0;
		for (int i = 1; i < label_row; i++) {
			if (label_.at<float>(i, 0) > mx_score) {
				mx_score = label_.at<float>(i, 0);
				label_index = i;
			}
		}
		cout << "Predict index: " << predict_index << endl;
		cout << "Label index: " << label_index << endl;
		cout << "Loss: " << loss << endl;
	}
	else {
		float sum_loss = 0;
		int num = 0;
		for (int i = 0; i < col; i++) {
			layer[0] = input.col(i);
			int predict_index = Predict1(layer[0]);
			sum_loss += loss;
			Mat label__ = label_.col(i);
			int label_index = 0;
			float mx_score = 0;
			//printf("%d %d %d\n", label_.rows, label_.cols, label_.channels());
			for (int k = 1; k < label_row; k++) {
				if (label__.at<float>(k, 0) > mx_score) {
					mx_score = label__.at<float>(k, 0);
					label_index = k;
				}
			}
			cout << "Test sample: " << i << "   " << "Predict: " << predict_index << endl;
			cout << "Test sample: " << i << "   " << "Label:  " << label_index << endl << endl;
			num += (predict_index == label_index);
		}
		accuracy = (float)num / col;
		float average_loss = sum_loss / col;
		std::cout << "Loss average: " << average_loss << std::endl;
		std::cout << "accuracy: " << accuracy << std::endl;
	}
}

int Net::Predict1(Mat input) {
	int row = input.rows;
	int col = input.cols;
	if (row == 0 || col == 0) {
		fprintf(stderr, "Input Is Empty!");
		exit(-1);
	}
	layer[0] = input;
	forward();
	int Size = layer.size() - 1;
	Mat output = layer[Size];
	float mxscore = output.at<float>(0, 0);
	int index = 0;
	for (int i = 1; i < output.rows; i++) {
		if (output.at<float>(i, 0) > mxscore) {
			mxscore = output.at<float>(i, 0);
			index = i;
		}
	}
	return index;
}

void Net::save_model(string file_name) {
	FileStorage model(file_name, FileStorage::WRITE);
	model << "layer_neuron_num" << layer_neuron_num;
	model << "learning_rate" << learning_rate;
	model << "activation_func" << activation_func;
	for (int i = 0; i < weights.size(); i++) {
		string weight = "weight_" + to_string(i);
		model << weight << weights[i];
	}
	model.release();
}

void Net::load_model(string file_name) {
	FileStorage fs;
	fs.open(file_name, FileStorage::READ);
	fs["layer_neuron_num"] >> layer_neuron_num;
	initNet(layer_neuron_num);
	for (int i = 0; i < weights.size(); i++) {
		string weight = "weight_" + to_string(i);
		fs[weight] >> weights[i];
	}
	fs["learning_rate"] >> learning_rate;
	fs["activation_func"] >> activation_func;
	fs.release();
}

void Net::loadData(string filename, Mat &INPUT, Mat &LABEL, int st, int sample_num) {
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	Mat input_, label_;
	fs["input"] >> input_;
	fs["label"] >> label_;
	fs.release();
	INPUT = input_(Rect(st, 0, sample_num, input_.rows));
	LABEL = label_(Rect(st, 0, sample_num, label_.rows));
}
