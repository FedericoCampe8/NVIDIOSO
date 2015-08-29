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
InDomainSearchInitializer::initialize_back ()
{
	// If first initialization: use initialize method
	if ( _initialized_values.size () == 0 )
	{
		initialize ();
		return;
	}
	
	// Use init values
	int value;
	for ( auto& var : _fd_variables )
	{
		// Sanity check
		if ( var->is_empty () )
		{
			std::string err_msg {"InDomainSearchInitializer::initialize_back - empty domain cannot initialize."};
        	throw NvdException ( err_msg.c_str() );
		}
		
		// Skip singleton variables
		if ( var->is_singleton () )
		{
			_initialized_values    [ var->get_id () ] = value;
			_initialized_variables [ var->get_id () ] = true;
			continue;
		}
		
		auto it = _initialized_values.find ( var->get_id () );
		if ( it == _initialized_values.end () )
		{
			std::string err_msg {"InDomainSearchInitializer::initialize_back - variable id not found."};
        	throw NvdException ( err_msg.c_str() );
		}
		
		value = _initialized_values [ var->get_id () ];
		
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
        	std::string err_msg {"InDomainSearchInitializer::initialize - variable not numeric."};
        	throw NvdException ( err_msg.c_str() );
        }
        _initialized_variables [ var->get_id () ] = true;
	}
}//initialize_back

void 
InDomainSearchInitializer::initialize ()
{
	int value;
	for ( auto& var : _fd_variables )
	{
		// Sanity check
		if ( var->is_empty () )
		{
			std::string err_msg {"InDomainSearchInitializer::initialize - empty domain cannot initialize."};
        	throw NvdException ( err_msg.c_str() );
		}
		
		// Skip singleton variables
		if ( var->is_singleton () )
		{
			_initialized_values    [ var->get_id () ] = value;
			_initialized_variables [ var->get_id () ] = true;
			continue;
		}
		
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
        	std::string err_msg {"InDomainSearchInitializer::initialize - variable not numeric."};
        	throw NvdException ( err_msg.c_str() );
        }
        
        _initialized_values    [ var->get_id () ] = value;
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