#include "Network.h"
#include "config.h"

int main() {
	Mat test_input, test_label;
	int sample_num = 200;
	int st = 800;

	Net net;
	net.load_model("F:\\make_data\\sigmoid_800_200.xml");
	net.loadData("F:\\make_data\\input_1000.xml", test_input, test_label, st, sample_num);
	net.Test(test_input, test_label);
	system("pause");
	return 0;
}