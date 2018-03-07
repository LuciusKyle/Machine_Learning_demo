
//
//									Hello World!
//------------------------------------------------------------------------------------------------
//
//						this is a cpp source file.
//
//						Lucius@LUCIUS-PC
//											--3/3/2018 07:04:23
//
//
//			Copyright (c) 2018 LuciusKyle@outlook.com. All rights reserved.
//
//------------------------------------------------------------------------------------------------
//									Goodbye World!
//


#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define  _CRT_SECURE_NO_WARNINGS
#endif // !_CRT_SECURE_NO_WARNINGS
#ifndef _SCL_SECURE_NO_WARNINGS
#define  _SCL_SECURE_NO_WARNINGS
#endif // !_SCL_SECURE_NO_WARNINGS
#endif // _MSC_VER

//#include"dll_Header.h"
#include"Neural_Network.h"

#include<random>
#include<fstream>
#include<string>
#include<ctime>	//use time(nullptr) as random seed;

#include <iostream>
#include<boost/numeric/ublas/io.hpp>


Neural_Network::Neural_Network()
	: weight_matrix_of_first_layer(16, 28 * 28)
	, weight_matrix_of_second_layer(16, 16)
	, weight_matrix_of_third_layer(10, 16)
	, bias_1(16, 1)
	, bias_2(16, 1)
	, bias_3(10, 1)
	, image_matrix(28 * 28, 1)

	, first_out_put(16, 1)
	, second_out_put(16, 1)
	, final_out_put(10, 1)
{

}

Neural_Network::Neural_Network(const Neural_Network &scr)
{
	this->weight_matrix_of_first_layer = scr.weight_matrix_of_first_layer;
	this->weight_matrix_of_second_layer = scr.weight_matrix_of_second_layer;
	this->weight_matrix_of_third_layer = scr.weight_matrix_of_third_layer;
	this->bias_1 = scr.bias_1;
	this->bias_2 = scr.bias_2;
	this->bias_3 = scr.bias_3;
	this->first_out_put = scr.first_out_put;
	this->second_out_put = scr.second_out_put;
	this->final_out_put = scr.final_out_put;
	this->image_matrix = scr.image_matrix;
}

void Neural_Network::init_all_matrix(bool random_init /*= true*/)
{
	if (random_init) {
		random_init_matrix(weight_matrix_of_first_layer);
		random_init_matrix(weight_matrix_of_second_layer);
		random_init_matrix(weight_matrix_of_third_layer);
		random_init_matrix(bias_1);
		random_init_matrix(bias_2);
		random_init_matrix(bias_3);
	}
	else {
		char *buf = new char[256];
		std::ifstream *file(new std::ifstream("matrixes"));
		//for(size_t i = 0;i<)
		auto single_operation = [file, buf](auto &mat)->void { //C++14 new feature.
			for (size_t i = 0; i < mat.size1(); ++i) {
				for (size_t ii = 0; ii < mat.size2(); ++ii) {
					file->getline(buf, 256);
					mat(i, ii) = std::strtod(buf, nullptr);
				}
			}
		};

		single_operation(weight_matrix_of_first_layer);
		single_operation(weight_matrix_of_second_layer);
		single_operation(weight_matrix_of_third_layer);
		single_operation(bias_1);
		single_operation(bias_2);
		single_operation(bias_3);

		delete buf;
		delete file;
	}
}

void Neural_Network::save_network_to_file()
{
	std::ofstream *file(new std::ofstream("matrixes"));
	file->clear();
	file->flush();

	auto single_operation = [file](const auto &mat)->void { //C++14 new feature.
		for (size_t i = 0; i < mat.size1(); ++i) {
			for (size_t ii = 0; ii < mat.size2(); ++ii) {
				(*file) << std::to_string(mat(i, ii)).append(1, '\n');
			}
		}
	};

	single_operation(weight_matrix_of_first_layer);
	single_operation(weight_matrix_of_second_layer);
	single_operation(weight_matrix_of_third_layer);
	single_operation(bias_1);
	single_operation(bias_2);
	single_operation(bias_3);

	file->flush();
	delete file;
}

void Neural_Network::random_init_matrix(boost::numeric::ublas::matrix<double> &mat)
{
	std::default_random_engine r_engine(time(nullptr));
	std::normal_distribution<double> Gaussian_distribution(0, 1);

	for (size_t i = 0; i < mat.size1(); ++i) {
		for (size_t ii = 0; ii < mat.size2(); ++ii) {
			mat(i, ii) = Gaussian_distribution(r_engine);
		}
	}
}

void Neural_Network::sigmoid(boost::numeric::ublas::matrix<double> &mat)
{
	for (size_t i = 0; i < mat.size1(); ++i) {
		for (size_t ii = 0; ii < mat.size2(); ++ii) {
			mat.at_element(i, ii) = sigmoid(mat.at_element(i, ii));
		}
	}
}

void Neural_Network::train_neural_network(const unsigned char *image_array, const int *value, const size_t data_count/* = 1*/, const double learning_rate/* = 0.1*/)
{
	using namespace boost::numeric::ublas;

	matrix<double>
		delta_final(final_out_put.size1(), 1, 0.0)
		, nabla_w3(delta_final.size1(), second_out_put.size1(), 0.0)
		, delta_second(second_out_put.size1(), 1, 0.0)
		, nabla_w2(delta_second.size1(), first_out_put.size1(), 0.0)
		, delta_first (first_out_put.size1(), 1, 0.0)
		, nabla_w1(delta_first.size1(), image_matrix.size1(), 0.0);

	for (size_t i = 0; i < data_count; ++i) {
		backpropagation(image_array + (i * 28 * 28), value[i]
			, delta_final
			, nabla_w3
			, delta_second
			, nabla_w2
			, delta_first
			, nabla_w1);
	}

	weight_matrix_of_first_layer -= ((learning_rate / data_count) * nabla_w1);
	weight_matrix_of_second_layer -= ((learning_rate / data_count) * nabla_w2);
	weight_matrix_of_third_layer -= ((learning_rate / data_count) * nabla_w3);
	bias_1 -= ((learning_rate / data_count) * delta_first);
	bias_2 -= ((learning_rate / data_count) * delta_second);
	bias_3 -= ((learning_rate / data_count) * delta_final);













	for (size_t i = 0; i < image_matrix.size1(); ++i) { //this is a single column matrix
		image_matrix(i, 0) = image_array[i] * 1.0 / 255;
	}
	axpy_prod(weight_matrix_of_first_layer, image_matrix, first_out_put);
	first_out_put += bias_1;
	sigmoid(first_out_put);
	axpy_prod(weight_matrix_of_second_layer, first_out_put, second_out_put);
	second_out_put += bias_2;
	sigmoid(second_out_put);
	axpy_prod(weight_matrix_of_third_layer, second_out_put, final_out_put);
	final_out_put += bias_3;
	sigmoid(final_out_put);

	std::vector<double>vec;
	for (size_t i = 0; i < final_out_put.size1(); ++i) {
		vec.push_back(final_out_put(i, 0));
	}
	double confidential = -1;
	int val = -1;
	for (size_t i = 0; i < vec.size(); ++i) {
		if (confidential < vec[i]) {
			confidential = vec[i];
			val = i;
		}
	}
	std::cout << "picture: " << value[0] << " net: " << val << " confidential: " << confidential << (value[0] == val ? "\tright" : "\twrong") << std::endl;
}


void Neural_Network::backpropagation(const unsigned char *image_array, const int value
	, boost::numeric::ublas::matrix<double> &total_delta_final
	, boost::numeric::ublas::matrix<double> &total_nabla_w3
	, boost::numeric::ublas::matrix<double> &total_delta_second
	, boost::numeric::ublas::matrix<double> &total_nabla_w2
	, boost::numeric::ublas::matrix<double> &total_delta_first
	, boost::numeric::ublas::matrix<double> &total_nabla_w1
)
{
	using namespace boost::numeric::ublas;
	for (size_t i = 0; i < image_matrix.size1(); ++i) { //this is a single column matrix
		image_matrix(i, 0) = image_array[i] * 1.0 / 255;
	}

	axpy_prod(weight_matrix_of_first_layer, image_matrix, first_out_put);
	auto z1 = first_out_put += bias_1;
	sigmoid(first_out_put);

	axpy_prod(weight_matrix_of_second_layer, first_out_put, second_out_put);
	auto z2 = second_out_put += bias_2;
	sigmoid(second_out_put);

	axpy_prod(weight_matrix_of_third_layer, second_out_put, final_out_put);
	auto z3 = final_out_put += bias_3;
	sigmoid(final_out_put);

	auto delta_final = matrix<double>(final_out_put.size1(), 1, 0.0);
	for (size_t i = 0; i < delta_final.size1(); ++i) {
		delta_final(i, 0) = (final_out_put(i, 0) - (value == i ? 0.9999 : 0.0001)) * sigmoid_prime(z3(i, 0));
	}
	matrix<double> nabla_w3(delta_final.size1(), second_out_put.size1(), 0.0);
	axpy_prod(delta_final, transpose(second_out_put), nabla_w3);

	auto delta_second = matrix<double>(second_out_put.size1(), 1, 0.0);
	axpy_prod(transpose(weight_matrix_of_third_layer), delta_final, delta_second);
	for (size_t i = 0; i < delta_second.size1(); ++i) {
		delta_second(i, 0) *= sigmoid_prime(z2(i, 0));
	}
	matrix<double> nabla_w2(delta_second.size1(), first_out_put.size1(), 0.0);
	axpy_prod(delta_second, transpose(first_out_put), nabla_w2);

	auto delta_first = matrix<double>(first_out_put.size1(), 1, 0.0);
	axpy_prod(transpose(weight_matrix_of_second_layer), delta_second, delta_first);
	for (size_t i = 0; i < delta_first.size1(); ++i) {
		delta_first(i, 0) *= sigmoid_prime(z1(i, 0));
	}
	matrix<double> nabla_w1(delta_first.size1(), image_matrix.size1(), 0.0);
	axpy_prod(delta_first, transpose(image_matrix), nabla_w1);

	total_delta_final += delta_final;
	total_nabla_w3 += nabla_w3;
	total_delta_second += delta_second;
	total_nabla_w2 += nabla_w2;
	total_delta_first += delta_first;
	total_nabla_w1 += nabla_w1;
}

const boost::numeric::ublas::matrix<double> Neural_Network::transpose(
	const boost::numeric::ublas::matrix<double> &input)
{
	auto ret = boost::numeric::ublas::matrix<double>(input.size2(), input.size1());
	for (size_t i = 0; i < input.size1(); ++i) {
		for (size_t ii = 0; ii < input.size2(); ++ii) {
			ret(ii, i) = input(i, ii);
		}
	}
	return ret;
}


bool Neural_Network::validate_neural_network(const unsigned char *image_array, int &value, double &confidential)
{
	confidential = -1.0;
	using namespace boost::numeric::ublas;
	for (size_t i = 0; i < image_matrix.size1(); ++i) { //this is a single column matrix
		image_matrix(i, 0) = image_array[i] * 1.0 / 255;
	}

	axpy_prod(weight_matrix_of_first_layer, image_matrix, first_out_put);
	first_out_put += bias_1;
	sigmoid(first_out_put);
	axpy_prod(weight_matrix_of_second_layer, first_out_put, second_out_put);
	second_out_put += bias_2;
	sigmoid(second_out_put);
	axpy_prod(weight_matrix_of_third_layer, second_out_put, final_out_put);
	final_out_put += bias_3;
	sigmoid(final_out_put);

	int calc_val = -1;
	for (size_t i = 0; i < final_out_put.size1(); ++i) {
		if (confidential < final_out_put(i, 0)) {
			calc_val = i;
			confidential = final_out_put(i, 0);
		}
	}

	if (calc_val == value) {
		return true;
	}
	else {
		value = calc_val;
		return false;
	}
}

