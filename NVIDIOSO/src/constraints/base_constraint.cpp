//
//  base_constraint.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/28/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "base_constraint.h"

BaseConstraint::BaseConstraint ( std::string name ) :
	Constraint () {
  _dbg             		= name + " - ";
  _scope_size      		= 0;
  _str_id          		= name;
  _number_id       		= (int) BaseConstraintType::OTHER;
  _base_constraint_type = BaseConstraintType::OTHER;
}//BaseConstraint

BaseConstraint::~BaseConstraint () {
}//~BaseConstraint

void 
BaseConstraint::set_base_constraint_type ( BaseConstraintType bse_t )
{
	_base_constraint_type = bse_t;
	_number_id = (int) _base_constraint_type;
}//set_base_constraint_type

BaseConstraintType 
BaseConstraint::get_base_constraint_type () const
{
	//Sanity check
	assert ( _base_constraint_type != BaseConstraintType::OTHER );
	return _base_constraint_type;
}//get_base_constraint_type

void 
BaseConstraint::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
	_scope = vars;
	for ( auto& s : args )
	{
		_arguments.push_back ( atoi ( s.c_str() ) );
	}
}//setup

void
BaseConstraint::attach_me_to_vars () {
  for ( auto var : scope() )
    var->attach_constraint( get_this_shared_ptr() );
}//attach_me_to_vars

int
BaseConstraint::unsat_level () const 
{
	throw NvdException ( (_dbg + "Constraint " + _str_id + " unsat_level not yet implemented").c_str() );
}//unsat_level

void
BaseConstraint::consistency () 
{
  throw NvdException ( (_dbg + "Constraint " + _str_id + " consistency not yet implemented").c_str() );
}//consistency

bool
BaseConstraint::satisfied () 
{
  throw NvdException ( (_dbg + "Constraint " + _str_id + " satisfied not yet implemented").c_str() );
}//satisfied

void
BaseConstraint::remove_constraint () {
  for ( auto var : scope() ) 
  {
	var->detach_constraint ( get_unique_id() );
  }
}//remove_constraint

void
BaseConstraint::print () const {
  std::cout << "Constraint_" << get_unique_id () << ": " << _str_id <<
  "\t (Base constraint id: " << _number_id << ")\n";
  std::cout << "Scope size: " << get_scope_size() << "\n";
  std::cout << "Weight:     " << get_weight ()     << "\n";
  if ( get_scope_size() ) {
  std::cout << "Variables:\n";
    for ( auto var : scope() ) std::cout << var->get_str_id() << " ";
    std::cout << std::endl;
  }
  if ( _arguments.size() ) {
    std::cout << "Arguments:\n";
    for ( auto arg : _arguments ) std::cout << arg << " ";
    std::cout << std::endl;
  }
}//print

void
BaseConstraint::print_semantic () const 
{
  std::cout << "Semantic:\n";
}//print_semantic
