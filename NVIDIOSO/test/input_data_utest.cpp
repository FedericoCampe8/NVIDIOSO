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
UnitTest("InputData") {}

InputDataUTest::~InputDataUTest() {}

bool
InputDataUTest::test()
{
	/*
	 * @note InputData is a singleton class which makes it difficult to test it.
	 *       This class was tested by re-building the module.
	 *       The following is a default unit test.
	 */
	int argc = 4; 
	char ** argv = (char**)malloc((argc + 1) * sizeof(char*));
	argv[0] = const_cast<char *> (std::string("program_name").c_str());
	argv[1] = const_cast<char *> (std::string("-i").c_str());
	argv[2] = const_cast<char *> (std::string("sample_input.fzn").c_str());
	argv[3] = const_cast<char *> (std::string("-v").c_str());

	// InputData instance
	InputData& idt = InputData::get_instance( argc, argv );
	
	// Unit Test
	TEST_TRUE (idt.verbose(), "verbose"); 
	TEST_FALSE(idt.timer(), "timer");
	TEST_EQUAL(idt.timeout(), -1.0, "timeout");
	TEST_EQUAL(idt.max_n_sol(), 1, "max_n_sol");
	TEST_EQUAL(idt.get_in_file(), std::string ("sample_input.fzn"), "get_in_file [test 1]");
	TEST_EQUAL(idt.get_out_file(), std::string (""), "get_out_file [test 1]");
	
	idt.set_input_file ( "test_in" );
	idt.set_output_file ( "test_out" );
	TEST_EQUAL(idt.get_in_file(), std::string ("test_in"), "get_in_file [test 2]");
	TEST_EQUAL(idt.get_out_file(), std::string ("test_out"), "get_out_file [test 2]");
	
	free (argv);
	
	return true;
}//run_test
