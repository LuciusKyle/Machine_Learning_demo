
//
//									Hello World!
//------------------------------------------------------------------------------------------------
//
//						this is a cpp source file.
//
//						Lucius@LUCIUS-PC
//											--3/2/2018 12:44:48
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
#endif // _MSC_VER

//#include"dll_Header.h"
#include"load_MNIST_database.h"

#include<fstream>

using std::string;

constexpr int high_or_low_endian_test = 0x12345678;

union Int_Char_convertor
{
	int i;
	unsigned char c[4];
};

load_MNIST_database::load_MNIST_database(string lable_file_path, string image_file_path)
	:p_labels(fopen(lable_file_path.c_str(), "rb"))
	, p_images(fopen(image_file_path.c_str(), "rb"))
	, number_of_items(-1)
{
	if (p_labels == NULL || p_images == NULL) {
		number_of_items = -1;
		return;
	}

	if (2049 != read_int(p_labels) || 2051 != read_int(p_images)) {
		number_of_items = -1;
		return;
	}

	int number_of_lables = read_int(p_labels);
	if (number_of_lables == read_int(p_images)) {
		number_of_items = number_of_lables;
		row_pixels = read_int(p_images);
		column_pixels = read_int(p_images);
	}
	else {
		number_of_items = -1;
	}
}

load_MNIST_database::~load_MNIST_database()
{
	if (p_labels != NULL) {
		fclose(p_labels);
	}
	if (p_images != NULL) {
		fclose(p_images);
	}
}

int load_MNIST_database::get_number_of_items() const
{
	return number_of_items;
}

int load_MNIST_database::read_int(FILE *p)
{
	int variable = -1;
	int start = 3;
	if (const char *test = (char*)&high_or_low_endian_test; *test == 0x78) {
		// do nothing;
	}
	else {
		start = 0;
		variable = 1;
	}
	int end = start + variable * 4;

	Int_Char_convertor my_union;
	for (int i = start; i != end; i += variable) {
		fread(&my_union.c[i], sizeof(decltype(my_union.c[i])), sizeof(my_union.c[i]) / sizeof(decltype(my_union.c[i])), p);
	}

	return my_union.i;
}

int load_MNIST_database::get_value(const size_t index) const
{
	if (0 != fseek(p_labels, sizeof(int) * 2 + sizeof(char) * index, SEEK_SET)) {
		return -1;
	}
	
	char buf;
	memset(&buf, 0, sizeof(decltype(buf)));
	fread(&buf, sizeof(decltype(buf)), sizeof(buf) / sizeof(decltype(buf)), p_labels);

	return buf;
}

int load_MNIST_database::get_pixel(const size_t index, unsigned char *array_of_char, size_t item_count/* = 1*/) const
{
	if (0 != fseek(p_images, sizeof(int) * 4 + sizeof(unsigned char) * index * 28 * 28, SEEK_SET)) {
		return -1;
	}

	if (row_pixels * column_pixels < 1) {
		return row_pixels * column_pixels;
	}

	unsigned char *buf = new unsigned char[row_pixels * column_pixels * item_count];

	fread(buf, row_pixels * column_pixels, sizeof(unsigned char) * item_count, p_images);
	array_of_char[0] = buf[0];
	for (size_t i = 0; i < row_pixels * column_pixels * item_count; ++i) {
		array_of_char[i] = buf[i];
	}

	delete buf;
	return row_pixels * column_pixels;
}



