//
//  simple_search_initializer.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/25/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_search_initializer.h"

using namespace std;

SimpleSearchInitializer::SimpleSearchInitializer ( std::vector< Variable* > vars ) :
	_dbg ( "SimpleSearchInitializer - " ) {
	_fd_variables = vars;
	
	// Init mapping 
	for ( auto& var : _fd_variables )
	{
		_initialized_variables [ var->get_id () ] = false;
	}
}//SimpleSearchInitializer

SimpleSearchInitializer::~SimpleSearchInitializer () {
}//~SimpleSearchInitializer

void 
SimpleSearchInitializer::set_variables ( std::vector < Variable* > vars )
{
	_fd_variables = vars;
	
	// Init mapping 
	_initialized_variables.clear ();
	for ( auto& var : _fd_variables )
	{
		_initialized_variables [ var->get_id () ] = false;
	}
}//set_variables
	
bool 
SimpleSearchInitializer::is_being_initializer ( Variable * var ) const
{
	// Sanity check
	assert ( var != nullptr );
	
	auto it = _initialized_variables.find ( var->get_id () );
	if ( it == _initialized_variables.end() )
	{
		return false;
	}
	return true;
}//is_being_initializer

bool 
SimpleSearchInitializer::is_initialized ( Variable * var ) const
{
	// Sanity check
	assert ( var != nullptr );
	
	auto it = _initialized_variables.find ( var->get_id () );
	if ( it == _initialized_variables.end() )
	{
		return false;
	}
	return _initialized_variables.at ( var->get_id () );
}//is_initialized

void 
SimpleSearchInitializer::print_initialization () const
{
	cout << _dbg << "Variables initialized as:\n";
	cout << "||";
	for ( auto& var : _fd_variables )
	{
		if ( _initialized_variables.at ( var->get_id () ) )
		{
			cout << " V_" << var->get_id () << " = {" << 
			initialization_value<int> ( var ) 
			<< "} |";
		}
	}
	cout << "|\n";
}//print_initilization