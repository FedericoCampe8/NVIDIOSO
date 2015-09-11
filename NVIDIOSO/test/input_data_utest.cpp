//
//  input_data_utest.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/10/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.d.
//

#include "input_data_utest.h"

using std::cout;
using std::endl;

InputDataUTest::InputDataUTest() :
UnitTest( "InputData" ) {}


InputDataUTest::~InputDataUTest() {}

bool
InputDataUTest::run_test()
{
	_failure_string = "Test failure";
	return false;
}//run_test

std::string  
InputDataUTest::get_failure () 
{
	return _failure_string;
}
