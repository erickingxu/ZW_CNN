#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main() {
	CvMLData train_data;
	train_data.read_csv("F:\\train.csv");
	Mat data = Mat(train_data.get_values(), true);
	if (!data.empty()) {
		cout << "Read CSV FILE SUCCESS!" << endl;
	}
	else {
		fprintf(stderr, "READ CSV FILE FAILED!");
	}
	cout << "row: " << data.rows << endl;
	cout << "col: " << data.cols << endl;
	cout << "channels: " << data.channels() << endl;
	Mat input = data(Rect(1, 1, 784, data.rows - 1)).t();
	Mat label = data(Rect(0, 1, 1, data.rows - 1));
	Mat label2(10, input.cols, CV_32F, Scalar::all(0));

	Mat input_normalized(input.size(), input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			input_normalized.at<float>(i, j) = input.at<float>(i, j) / 255.0;
		}
	}

	for (int i = 0; i < label.rows; i++) {
		float index = label.at<float>(i, 0);
		label2.at<float>(index, i) = index;
	}

	string filename = "input.xml";
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "input" << input_normalized;
	fs << "label" << label2;
	fs.release();

	Mat input_1000 = input_normalized(Rect(0, 0, 1000, input_normalized.rows));
	Mat label_1000 = label2(Rect(0, 0, 1000, label2.rows));
	
	filename = "input_1000.xml";
	FileStorage fs2(filename, FileStorage::WRITE);
	fs2 << "input" << input_1000;
	fs2 << "label" << label2; // Write cv::Mat
	fs2.release();
	system("pause");
	return 0;
}