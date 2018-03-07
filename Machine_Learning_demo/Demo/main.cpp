
//
//									Hello World!
//------------------------------------------------------------------------------------------------
//
//						this is a cpp source file.
//
//						Lucius@LUCIUS-PC
//											--3/1/2018 03:00:14
//
//
//			Copyright (c) 2018 LuciusKyle@outlook.com. All rights reserved.
//
//------------------------------------------------------------------------------------------------
//									Goodbye World!
//


#include<iostream>
#include<fstream>
#include<vector>
//#include<boost/numeric/ublas/matrix.hpp>
//#include<boost/numeric/ublas/operation.hpp>

#include"../load_MNIST_database/load_MNIST_database.h"
#include"../Neural_Network/Neural_Network.h"

#ifdef _MSC_VER


#ifdef _WIN64
#ifdef _DEBUG
#pragma comment(lib, "../x64/Debug/load_MNIST_database.lib")
#pragma comment(lib, "../x64/Debug/Neural_Network.lib")
#else // _DEBUG
#pragma comment(lib, "../x64/Release/load_MNIST_database.lib")
#pragma comment(lib, "../x64/Release/Neural_Network.lib")
#endif // _DEBUG
#else // _WIN64
#ifdef _DEBUG
#pragma comment(lib, "../Debug/load_MNIST_database.lib")
#pragma comment(lib, "../Debug/Neural_Network.lib")
#else
#pragma comment(lib, "../Release/load_MNIST_database.lib")
#pragma comment(lib, "../Release/Neural_Network.lib")
#endif // _DEBUG
#endif // _WIN64

#endif // _MSC_VER

using std::string;
using std::vector;

constexpr size_t mini_batch = 1;
constexpr double learning_rate = 0.3;

int main(int argc, char *argv[])
{
	Neural_Network *network(new Neural_Network);
	network->init_all_matrix();

	load_MNIST_database training_data(
		"E:/code/THE MNIST DATABASE of handwritten digits/train-labels.idx1-ubyte"
		, "E:/code/THE MNIST DATABASE of handwritten digits/train-images.idx3-ubyte"
	);

	const int item_count = training_data.get_number_of_items();
	unsigned char *pixels = new unsigned char[28 * 28 * mini_batch];

	int *val = new int[mini_batch];
	for (int i = 0; i < item_count / mini_batch; ++i) {
		for (size_t ii = 0; ii < mini_batch; ++ii) {
			val[ii] = training_data.get_value(i + ii);
		}
		training_data.get_pixel(i * mini_batch, pixels, mini_batch);
		network->train_neural_network(pixels, val, mini_batch, learning_rate);
	}

	network->save_network_to_file();
	load_MNIST_database test_data(
		"E:/code/THE MNIST DATABASE of handwritten digits/t10k-labels.idx1-ubyte"
		, "E:/code/THE MNIST DATABASE of handwritten digits/t10k-images.idx3-ubyte"
	);
	const int test_item_count = test_data.get_number_of_items();

	Neural_Network validation_network;
	validation_network.init_all_matrix(false);
	size_t correct_count = 0;
	for (int i = 0; i < test_item_count; ++i) {
		test_data.get_pixel(i, pixels);
		int real_val = test_data.get_value(i);
		double confidential = 0;
		bool correct = validation_network.validate_neural_network(pixels, real_val, confidential);
		if (correct) {
			++correct_count;
		}
	}
	std::cout << "accuracy is: " << (100.0 * correct_count / test_item_count) << '%' << std::endl;

	delete val;
	delete pixels;
	delete network;

	return 0;
}


