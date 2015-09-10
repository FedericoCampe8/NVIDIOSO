//
//  unit_test_register.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "unit_test_register.h"
	
using namespace std;

// ---------------------------------------------------------------------- //
// -------------- POSTERs FUNCTIONs FOR GLOBAL CONSTRAINTS -------------- //
// ---------------------------------------------------------------------- //

/*
UnitTest* p_input_data_utest () 
{
	return new InputDataUTest ( "InputData" );
}//input_data_utest
*/

// ---------------------------------------------------------------------- //
// ---------------------------------------------------------------------- //

UnitTestRegister::UnitTestRegister () {
	fill_register ();
}//UnitTestRegister

UnitTestRegister::~UnitTestRegister () {
}//~UnitTestRegister

void 
UnitTestRegister::add ( std::string test_name, utest_poster p )
{
	_register [ test_name ] = p;
}//add 
 
UnitTestSPtr 
UnitTestRegister::get_unit_test ( std::string unit_test_name )
{
	auto it = _register.find ( unit_test_name );
	if ( it == _register.end () )
	{
		LogMsg << "UnitTestRegister::get_unit_test - " << unit_test_name << " not found." << endl;
		return nullptr;
	}
	
	// Create a new unit test instance using the poster
	UnitTestSPtr utest_ptr = shared_ptr<UnitTest> ( _register [ unit_test_name ] () );
	
	// Return the test instance instance
	return utest_ptr;
}//get_unit_test

std::vector< UnitTestSPtr > 
UnitTestRegister::get_unit_test ()
{
	std::vector< UnitTestSPtr > uTests;
	for ( auto& pr : _register )
	{
		uTests.push_back ( std::shared_ptr<UnitTest> ( (pr.second)() ) );
	}
	
	return uTests;
}//get_unit_test

void 
UnitTestRegister::fill_register ()
{
	//add ( "input_data", p_input_data_utest );
}//fill_register
