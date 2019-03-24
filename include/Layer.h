#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

namespace zwcnn {
	class Layer {
	public:
		Layer() {}
		virtual ~Layer();
		//层的名字
		string name;
		//层的类型
		string type;
		//当前层需要输入的blob的index
		vector <int> bottoms;
		//当前层产生的blob的index
		vector <int> top;
		//加载param
		virtual int load_param(FILE* p);
		//加载bin
		virtual int load_param_bin(FILE* p);
		virtual int forward(const vector <Mat> &bottom_blob, vector <Mat> &top_blob) const;
	};
	namespace LayerType {
		enum
		{
			AbsVal = 0,
			ArgMax = 1,
			BatchNorm = 2,
			Bias = 3,
			BNLL = 4,
			Concat = 5,
			Convolution = 6,
			Crop = 7,
			Deconvolution = 8,
			Dropout = 9,
			ELU = 10,
			Eltwise = 11,
			Embed = 12,
			Exp = 13,
			Flatten = 14,
			InnerProduct = 15,
			Input = 16,
			Log = 17,
			LRN = 18,
			MemoryData = 19,
			MVN = 20,
			Pooling = 21,
			Power = 22,
			PReLU = 23,
			Proposal = 24,
			Reduction = 25,
			ReLU = 26,
			Reshape = 27,
			ROIPooling = 28,
			Scale = 29,
			Sigmoid = 30,
			Slice = 31,
			Softmax = 32,
			Split = 33,
			SPP = 34,
			TanH = 35,
			Threshold = 36,
			Tile = 37,
			RNN = 38,
			LSTM = 39,
			BinaryOp = 40,
			UnaryOp = 41,

			CustomBit = (1 << 8),
		};
	}
	//layer factory function
	typedef Layer* (*layer_creator_func)();
	struct layer_registry_entry
	{
		// layer type name
		const char* name;
		// layer factory entry
		layer_creator_func creator;
	};
	//从layer的名字获得对应的Type类型
	int layer_to_index(const char* type);
	//创建新的layer
	Layer* create_layer(int index);
	#define DEFINE_LAYER_CREATOR(name) Layer* name##_layer_creator() { return new name; }
}