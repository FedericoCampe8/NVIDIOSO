//
//  unit_test.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Interface for unit test classes.
//  Unit test classes for iNVIDIOSO should implement this interface and register 
//  themself into the global array of unit test classes.
//  For every new unit test class, all previous unit test should be re-run.
//  @note Only header file, no .cpp for this class. 
//

#include "globals.h"

#ifndef __NVIDIOSO__unit_test__
#define __NVIDIOSO__unit_test__

class UnitTest;
typedef std::unique_ptr<UnitTest> UnitTestUPtr;
typedef std::shared_ptr<UnitTest> UnitTestSPtr;

class UnitTest {
protected:

	//! Unit test class 
	std::string _u_test_class;

public:
	/**
	* Constructor.
	* @param unit_test_class string describing the class the test belongs to.
	* @note example unit_test_class = "int_ne_constraint".
	*/
	UnitTest ( std::string unit_test_class ) :
		_u_test_class(unit_test_class) {}

	virtual ~UnitTest() {}

	//! Get name of the unit test class
	std::string get_unit_test_class_name () const
	{
		return _u_test_class;
	}//get_unit_test_class_name

	/**
	* This is the function running the unit test.
	* @return true if the test succeed, false otherwise.
	*/
	virtual bool run_test () = 0;

	/**
	 * Get a string describing the explanation why 
	 * the current test has failed.
	 * @return a string describing the failure.
	 */
	virtual std::string get_failure () = 0;
	
	//! Print information about this unit test
	virtual void print () const
	{
		std::cout << "Unit test class:\t" << _u_test_class << std::endl;
	}//print	
};

#endif
