//
//  unit_test_register.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Register class for unit test instances.
//  This class keeps an internal book of unit test instances.
//  Each new unit test class should register itself as a new entry 
//  in the book represented by the internal state of this register. 
//

#include "unit_test.h"

#ifndef __NVIDIOSO__unit_test_register__
#define __NVIDIOSO__unit_test_register__

typedef UnitTest* (*utest_poster) ();

class UnitTestRegister {
private:

	std::unordered_map<std::string, utest_poster> _register;

	/**
	* Add the poster (function to create the unit test instance ) p
	* to the map, with key name, i.e., the name of the unit test class.
	* @param name name of the unit test class.
	* @param p poster, function to instantiate a unit test class.
	*/
	void add(std::string test_name, utest_poster p);

	//! Init function, fill the register (hash table) with posters
	void fill_register();

	// Singleton instance: private constructor
	UnitTestRegister();

public:
	UnitTestRegister(const UnitTestRegister& other) = delete;
	UnitTestRegister& operator= (const UnitTestRegister& other) = delete;

	virtual ~UnitTestRegister();

	//! Constructor get (static) instance
	static UnitTestRegister& get_instance()
	{
		static UnitTestRegister register_instance;
		return register_instance;
	}//get_instance

	//! Given the string name of the global constraint, return an instance of it
	UnitTestSPtr get_unit_test(std::string unit_test_name);

	/*
	*  Get a vector of pointer to the instances of all the unit test classes
	*  currenlty registered to this register.
	*/
	std::vector< UnitTestSPtr > get_unit_test();
};

#endif	