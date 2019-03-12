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
	//ÿһ�����Ԫ��Ŀ
	vector <int> layer_neuron_num;
	//��ʼ������
	//0��ʾ[-1��1]�����ʼ��
	//1��ʾ��ֵΪGaussMean����ΪGaussSigma�ĸ�˹��ʼ��
	int InitType = 0;
	//��˹��ʼ���ķ���Ĭ��Ϊ1.0
	float GaussSigma = 1.0;
	//��˹��ʼ���ľ�ֵĬ��Ϊ0
	float GaussMean = 0;
	//openmp�߳���, Ĭ��Ϊ1
	//���������
	string activation_func = "sigmoid";
	//��ʧ��������
	string loss_type = "L1";
	//���򴫲�����
	void backward();
	//ѧϰ��
	float learning_rate = 0.001;
	//ѵ���ĵ�������
	int train_iter = 0;
	//���Ե����Ĵ���
	int test_iter = 0;
	//batch�Ĵ�С
	int batch_size = 1;
	//accuracy
	float accuracy = 0.0;
protected:
	//�㣬ʹ��Opencv��Mat
	vector <Mat> layer;
	//Ȩֵ����
	vector <Mat> weights;
	//ƫ�þ���
	vector <Mat> biases;
	//��ʧ���������в������ݶ�
	vector <Mat> delta;
	//ǰ�򴫲������һ�����������򴫲������
	Mat output_error;
	//�����ǩMat
	Mat label;
	//������ʧ������ֵ
	float loss = 0;
public:
	Net() {};
	~Net() {};
	//��ʼ������
	void initNet(vector <int> layer_neuron_num_);
	//������ͨ�������
	float *random_uniform(int length);
	//������˹�ֲ������
	float *random_gaussian(int length);
	//��ʼ��Ȩ��
	void initWeights();
	//��ʼ��ƫ��
	void initBiases();
	//��ʼ��ÿ��Layer���������
	void SetActivation(string input);
	//ǰ�򴫲�
	void forward();
	//ʵ��Tarin
	void Train(Mat input, Mat label_);
	//ʵ��Test
	void Test(Mat input, Mat label_);
	//ʵ��batchΪ1��predict��argmax
	int Predict1(Mat input);
	//����ѵ����ģ��
	void save_model(string file_name);
	//����ģ��
	void load_model(string file_name);
	//��������
	void loadData(string filename, Mat &INPUT, Mat &LABEL, int st, int sample_num);
protected:
	//���򴫲����ݶȣ���ʽ����
	void getGrad();
	//����ȥ��ȡ���ݶȺ�ѧϰ�ʲ�������
	void UpdateParameters();
};