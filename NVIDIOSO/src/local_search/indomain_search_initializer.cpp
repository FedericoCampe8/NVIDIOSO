//
//  indomain_search_initializer.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/25/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "indomain_search_initializer.h"
#include "int_variable.h"

using namespace std;

InDomainSearchInitializer::InDomainSearchInitializer ( std::vector< Variable* > vars, InDomainInitType indomain_t ) :
	SimpleSearchInitializer ( vars ) {
	_dbg = "InDomainSearchInitializer - ";
	_initialization_type = indomain_t;
}//InDomainSearchInitializer

InDomainSearchInitializer::~InDomainSearchInitializer () {
}//~InDomainSearchInitializer

void 
InDomainSearchInitializer::set_initialization_type ( InDomainInitType indom_t )
{
	_initialization_type = indom_t;
}//set_initialization_type

InDomainInitType
InDomainSearchInitializer::get_initialization_type () const
{
	return _initialization_type;
}//set_initialization_type

void 
InDomainSearchInitializer::initialize ()
{
	int value;
	for ( auto& var : _fd_variables )
	{
		if ( _initialization_type == InDomainInitType::INDOMAIN_MIN )
		{
			value = var->domain_iterator->min_val ();
		}
		else if ( _initialization_type == InDomainInitType::INDOMAIN_MAX )
		{
			value = var->domain_iterator->max_val ();
		}
		else if ( _initialization_type == InDomainInitType::INDOMAIN_RAN )
		{
			value = var->domain_iterator->random_val ();
		}
		else
		{
			value = var->domain_iterator->random_val ();
		}
		
		/*
       	 * Here it comes the actual labeling.
       	 * @note care must be taken for non int variables (e.g. set, float).
       	 * @note it automatically notifies the attached store.
       	 */
      	if ( var->domain_iterator->is_numeric () )
      	{
        	(static_cast<IntVariable*>(var))->shrink ( value, value );   
        }
        else
        {
        	std::string err_msg {"InDomainSearchInitializer::initialize variable not numeric"};
        	throw NvdException ( err_msg.c_str() );
        }
        
		_initialized_variables [ var->get_id () ] = true;
	}
}//initialize

void
InDomainSearchInitializer::print () const
{
	cout << "InDomain Search Initializer:\n";
	cout << "Variable (Id) - Initialized (Yes/No)\n";
	cout << "------------------------------------\n";
	for ( auto& var : _initialized_variables )
	{
		cout << "Variable (Id) - Initialized (Yes/No)\n";
		cout << "   V_" << var.first << " |    "  << std::boolalpha << var.second << endl;
	}
	cout << "------------------------------------\n";
}//print