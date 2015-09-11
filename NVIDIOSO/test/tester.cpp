//
//  tester.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "tester.h"

using std::cout;
using std::endl;

UnitTestRegister& utest_register = UnitTestRegister::get_instance();

Tester::Tester() {}

Tester::~Tester() {}

void
Tester::run()
{
	bool pass;
	std::string test_name{};
	std::string test_failure{};
	std::string line { "=============================" };
	
	LogMsg << " TEST NAME            PASSED" << std::endl;
	LogMsg << line << std::endl;
	std::vector< UnitTestSPtr > vec_test = utest_register.get_unit_test(); 
	for (auto& test : vec_test)
	{
		test_name = test->get_unit_test_class_name();
		LogMsg << "| " << test_name << ":";
		int wss = line.size () - (test_name.size () + 5);
		for ( ; wss >= 0; wss-- )  LogMsg << " ";
		LogMsg << "|" << std::endl;
		pass = test->run_test();
		if ( pass )
		{
			LogMsg << "|\t\t\tYES |" << std::endl;
		}
		else
		{
			test_failure = test->get_failure ();
			LogMsg << "|\t\t\tNO  |" << std::endl;
			break;
		}
	}
	LogMsg << line << std::endl;
	
	// Throw exception
	throw std::logic_error ( test_name + test_failure );
}//run

