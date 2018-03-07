
//
//									Hello World!
//------------------------------------------------------------------------------------------------
//
//						this is a cpp header file.
//
//						Lucius@LUCIUS-PC
//											--3/3/2018 07:04:06
//
//
//			Copyright (c) 2018 LuciusKyle@outlook.com. All rights reserved.
//
//------------------------------------------------------------------------------------------------
//									Goodbye World!
//


#ifndef _Neural_Network_H_
#define _Neural_Network_H_

#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define  _CRT_SECURE_NO_WARNINGS
#endif // !_CRT_SECURE_NO_WARNINGS
#endif // _MSC_VER

#include<boost/numeric/ublas/matrix.hpp>
#include<boost/numeric/ublas/operation.hpp>

#ifdef _MSC_VER
#ifdef NEURALNETWORK_EXPORTS
#define NEURALNETWORK_API __declspec(dllexport)
#else
#define NEURALNETWORK_API __declspec(dllimport)
#endif // NEURALNETWORK_EXPORTS
#endif // _MSC_VER


class NEURALNETWORK_API Neural_Network
{
public:
	Neural_Network();

	Neural_Network(const Neural_Network &scr);

	void init_all_matrix(bool random_init = true);
	void save_network_to_file();
	void train_neural_network(const unsigned char *image_array, const int *value, const size_t data_count = 1, const double learning_rate = 0.1);
	bool validate_neural_network(const unsigned char *image_array, int &value, double &confidential);
private:

	void random_init_matrix(boost::numeric::ublas::matrix<double> &);

	inline double sigmoid(double x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	inline double sigmoid_prime(const double z)
	{
		return sigmoid(z) * (1.0 - sigmoid(z));
	}

	void sigmoid(boost::numeric::ublas::matrix<double> &);

	void backpropagation(const unsigned char *image_array, const int value
		, boost::numeric::ublas::matrix<double> &
		, boost::numeric::ublas::matrix<double> &
		, boost::numeric::ublas::matrix<double> &
		, boost::numeric::ublas::matrix<double> &
		, boost::numeric::ublas::matrix<double> &
		, boost::numeric::ublas::matrix<double> &
	);

	const boost::numeric::ublas::matrix<double> transpose(
		const boost::numeric::ublas::matrix<double> &input
	);

	boost::numeric::ublas::matrix<double>
		weight_matrix_of_first_layer
		, weight_matrix_of_second_layer
		, weight_matrix_of_third_layer
		, bias_1
		, bias_2
		, bias_3;

	boost::numeric::ublas::matrix<double>
		first_out_put
		, second_out_put
		, final_out_put;

	boost::numeric::ublas::matrix<double>
		image_matrix;
};





#endif // !_Neural_Network_H_
