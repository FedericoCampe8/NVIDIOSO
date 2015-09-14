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

Tester::Tester() :
	_on_test ( "" ) {	
}

Tester::~Tester() {
}

void 
Tester::set_running_test ( std::string test_name )
{
	_on_test = test_name;
}//set_running_test

void
Tester::run()
{
	bool pass;
	std::string test_name{};
	std::string test_failure{};
	std::string line { "=============================" };
	
	LogMsgUT << " TEST NAME    PASSED (YES/NO)" << std::endl;
	LogMsgUT << line << std::endl;
	 
	if ( _on_test != "" )
	{// Run the specified test
		auto test = utest_register.get_unit_test ( _on_test );
		test_name = test->get_unit_test_class_name();
		
		LogMsgUT << "| " << test_name << ":" << std::endl;
		pass = test->run_test();
		if ( pass )
		{
			LogMsgUT << "|\t\t\tYES " << std::endl;
		}
		else
		{
			test_failure = test->get_failure ();
			LogMsgUT << "|\t\t\tNO  " << std::endl;
		}
	}
	else
	{// No test name set: run all tests
		std::vector< UnitTestSPtr > vec_test = utest_register.get_unit_test();
		for (auto& test : vec_test)
		{
			test_name = test->get_unit_test_class_name();
		
			LogMsgUT << "| " << test_name << ":" << std::endl;
			/*
		 	 * int wss = line.size () - (test_name.size () + 5);
		 	 * for ( ; wss >= 0; wss-- )  LogMsgUT << " ";
		 	 * LogMsgUT << "|" << std::endl;
		 	 */
			pass = test->run_test();
			if ( pass )
			{
				LogMsgUT << "|\t\t\tYES " << std::endl;
			}
			else
			{
				test_failure = test->get_failure ();
				LogMsgUT << "|\t\t\tNO  " << std::endl;
				break;
			}
		}
	}
	LogMsgUT << line << std::endl;
	
	// Throw exception
	if ( !pass )
	{
		throw std::logic_error ( test_name + " " + test_failure );
	}
}//run

