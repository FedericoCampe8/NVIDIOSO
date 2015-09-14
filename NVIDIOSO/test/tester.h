//
//  tester.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Tester class to run unit tests.
//  This class runs all the unit tests registered in the unit test register.
//

#include "globals.h"
#include "unit_test_register.h"

#ifndef __NVIDIOSO__tester__
#define __NVIDIOSO__tester__

extern UnitTestRegister& utest_register;

class Tester {
private:
	//! Unit test to run (default run all registered tests)
	std::string _on_test;	
	
public:
	Tester();

	virtual ~Tester();
	
	/**
	 * Set test to run.
	 * @param test_name string representing the name 
	 *        of the (registered) unit test to perform.
	 */ 
	void set_running_test ( std::string test_name );
	
	//! @note it throws for failed tests 
	virtual void run();
};

#endif