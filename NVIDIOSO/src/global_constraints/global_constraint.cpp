//
//  global_constraint.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/31/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "global_constraint.h"

GlobalConstraint::GlobalConstraint ( std::string name ) :
	Constraint () {
  	_global          = true;
  	_str_id          = name;
  	_dbg             = name + " - ";
  	_scope_size      = 0;
  	_num_blocks      = 1;
  	_num_threads     = 1;
  	_local_memory    = 0;
}//GlobalConstraint

GlobalConstraint::~GlobalConstraint () {
}//~FZNConstraint

void 
GlobalConstraint::naive_consistency ()
{
	throw NvdException ( (_dbg + "Constraint " + _str_id + " naive consistency not yet implemented").c_str() );
}//naive_consistency

void 
GlobalConstraint::bound_consistency ()
{
	throw NvdException ( (_dbg + "Constraint " + _str_id + " bound consistency not yet implemented").c_str() );
}//bound_consistency

void 
GlobalConstraint::full_consistency ()
{
	throw NvdException ( (_dbg + "Constraint " + _str_id + " full consistency not yet implemented").c_str() );
}//full_consistency

size_t 
GlobalConstraint::get_num_blocks () const
{
	return _num_blocks;
}//get_num_blocks

size_t 
GlobalConstraint::get_num_threads () const
{
	return _num_threads;
}//get_num_blocks

size_t 
GlobalConstraint::get_size_memory () const
{
	return _local_memory;
}//get_num_blocks

void 
GlobalConstraint::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

void
GlobalConstraint::attach_me_to_vars () 
{
	for ( auto var : scope() )
	{
    	var->attach_constraint( get_this_shared_ptr() );
    }
}//attach_me

void
GlobalConstraint::consistency () 
{
	switch ( _consistency )
  	{
  		case ConsistencyType::NAIVE_C:
  			naive_consistency ();
  			break;
  		case ConsistencyType::BOUND_C:
  			bound_consistency ();
  			break;
  		case ConsistencyType::DOMAIN_C:
  			full_consistency ();
  			break;
  		default:
  			naive_consistency ();
  			break;
  	}
}//consistency

bool
GlobalConstraint::satisfied () 
{
	throw NvdException ( (_dbg + "Constraint " + _str_id + " not yet implemented").c_str() );
}//satisfied

void
GlobalConstraint::remove_constraint () 
{
	for ( auto var : scope() ) 
  	{
		var->detach_constraint ( get_unique_id() );
  	}
}//remove_constraint

void
GlobalConstraint::print () const 
{
	std::cout << "Global constraint:\n";
	std::cout << "Constraint_" << get_unique_id () << ": " << _str_id << "\n";
  	std::cout << "Scope size: " << get_scope_size() << "\n";
  	std::cout << "Weight:     " << get_weight ()     << "\n";
  	std::cout << "Propagator: ";
  	switch ( _consistency )
  	{
  		case ConsistencyType::NAIVE_C:
  			std::cout << "Naive\n";
  			break;
  		case ConsistencyType::BOUND_C:
  			std::cout << "Bound\n";
  			break;
  		case ConsistencyType::DOMAIN_C:
  			std::cout << "Full\n";
  			break;
  		default:
  			std::cout << "Naive\n";
  			break;
  	}
  	if ( get_scope_size() ) 
  	{
  		std::cout << "Variables:\n";
    	for ( auto var : scope() ) std::cout << var->get_str_id() << " ";
    	std::cout << std::endl;
  	}
  	if ( _arguments.size() ) 
  	{
    	std::cout << "Arguments:\n";
   		for ( auto arg : _arguments ) std::cout << arg << " ";
    	std::cout << std::endl;
  	}
}//print

void
GlobalConstraint::print_semantic () const 
{
	std::cout << "Semantic:\n";
}//print_semantic
