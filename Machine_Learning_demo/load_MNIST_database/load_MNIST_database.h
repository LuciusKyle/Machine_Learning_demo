
//
//									Hello World!
//------------------------------------------------------------------------------------------------
//
//						this is a cpp header file.
//
//						Lucius@LUCIUS-PC
//											--3/2/2018 12:42:08
//
//
//			Copyright (c) 2018 LuciusKyle@outlook.com. All rights reserved.
//
//------------------------------------------------------------------------------------------------
//									Goodbye World!
//


#ifndef _load_MNIST_database_H_
#define _load_MNIST_database_H_

#ifdef _MSC_VER

#ifdef load_MNIST_database_EXPORTS
#define load_MNIST_database_API __declspec(dllexport)
#else
#define load_MNIST_database_API __declspec(dllimport)
#endif // load_MNIST_database_EXPORTS

#else

#define load_MNIST_database_EXPORTS

#endif // _MSC_VER

#include<string>
#include<iostream>

class load_MNIST_database_API load_MNIST_database
{
public:
	load_MNIST_database(const std::string lable_file_path, const std::string image_file_path);
	~load_MNIST_database();

	int get_number_of_items() const;

	int get_value(const size_t index) const; //return the value of that index;
	int get_pixel(const size_t index, unsigned char *array_of_char, size_t item_count = 1) const; //return pixel count of that index;

private:
	int read_int(FILE *p);
	int number_of_items;
	
	int row_pixels;
	int column_pixels;

	FILE * p_labels;
	FILE * p_images;
};


#endif // !_load_MNIST_database_H_
