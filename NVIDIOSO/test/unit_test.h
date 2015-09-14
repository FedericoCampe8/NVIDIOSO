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

#ifndef __NVIDIOSO__unit_test__
#define __NVIDIOSO__unit_test__

#include "globals.h"

class UnitTest;
typedef std::unique_ptr<UnitTest> UnitTestUPtr;
typedef std::shared_ptr<UnitTest> UnitTestSPtr;

class UnitTest {
private:
	//! Internal test state
	bool _test_success;

protected:

	//! Unit test class 
	std::string _u_test_class;

	//! Unit test failure string
	std::string _u_test_failure;

	void TEST_TRUE(bool b, std::string msg = "")
	{
		if (!_test_success) return;
		if (!b)
		{
			_u_test_failure = msg;
			_u_test_failure += " => on TEST_TRUE";
			_test_success = false;
		}
	}//TEST_TRUE

	void TEST_FALSE(bool b, std::string msg = "")
	{
		if (!_test_success) return;
		if (b)
		{
			_u_test_failure = msg;
			_u_test_failure += " => on TEST_FALSE";
			_test_success = false;
		}
	}//TEST_FALSE

	void TEST_NULL(void* ptr, std::string msg = "")
	{
		if (!_test_success) return;
		if (ptr != NULL)
		{
			_u_test_failure = msg;
			_u_test_failure += " => on TEST_NULL";
			_test_success = false;
		}
	}//TEST_NULL

	void TEST_NOT_NULL(void* ptr, std::string msg = "")
	{
		if (!_test_success) return;
		if (ptr == NULL)
		{
			_u_test_failure = msg;
			_u_test_failure += " => on TEST_NOT_NULL";
			_test_success = false;
		}
	}//TEST_NOT_NULL

	template<typename T>
	void TEST_EQUAL(T a, T b, std::string msg = "")
	{
		if (!_test_success) return;
		if (a != b)
		{
			_u_test_failure = msg;
			_u_test_failure += " => on TEST_EQUAL: ";
			std::ostringstream convert1; convert1 << a;
			std::ostringstream convert2; convert2 << b;
			_u_test_failure += convert1.str();
			_u_test_failure += " != ";
			_u_test_failure += convert2.str();
			_test_success = false;
		}
	}//TEST_EQUAL
	
	template<typename T>
	void TEST_NOT_EQUAL(T a, T b, std::string msg = "")
	{
		if (!_test_success) return;
		if (a == b)
		{
			_u_test_failure = msg;
			_u_test_failure += " => on TEST_NOT_EQUAL: ";
			std::ostringstream convert1; convert1 << a;
			std::ostringstream convert2; convert2 << b;
			_u_test_failure += convert1.str();
			_u_test_failure += " == ";
			_u_test_failure += convert2.str();
			_test_success = false;
		}
	}//TEST_NOT_EQUAL
	
	/*
	* Function which is class-specific.
	* This function can throw if a MACRO is not satisfied.
	* @return true if the test succeed, false otherwise.
	*/
	virtual bool test() = 0;

public:
	/**
	* Constructor.
	* @param unit_test_class string describing the class the test belongs to.
	* @note example unit_test_class = "int_ne_constraint".
	*/
	UnitTest(std::string unit_test_class) :
		_test_success(true),
		_u_test_class(unit_test_class) {}

	virtual ~UnitTest() {}

	//! Get name of the unit test class
	std::string get_unit_test_class_name() const
	{
		return _u_test_class;
	}//get_unit_test_class_name

	/**
	* This is the function running the unit test.
	* @return true if the test succeed, false otherwise.
	*/
	virtual bool run_test() 
	{
		_test_success &= test();
		return _test_success;
	}//run_test

	/**
	 * Set failure message to print on test failure.
	 * @param failure_msg failure message describing the unit test failure.
	 * @note This method set internal state to a failure state.
	 */
	virtual void set_failure_message ( std::string failure_msg )
	{
		_test_success = false;
		_u_test_failure = failure_msg;
	}//set_failure_message
	
	/**
	* Get a string describing the explanation why
	* the current test has failed.
	* @return a string describing the failure.
	*/
	virtual std::string get_failure()
	{
		if (_u_test_failure == "")
		{
			_u_test_failure = "No failure string available";
		}
		return _u_test_failure;
	}//get_failure
	
	//! Print information about this unit test
	virtual void print() const
	{
		std::cout << "Unit test class:\t" << _u_test_class << std::endl;
	}//print	
};

#endif
