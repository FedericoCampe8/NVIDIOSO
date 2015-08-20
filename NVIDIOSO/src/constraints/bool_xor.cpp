//
//  bool_xor.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_xor.h"

BoolXor::BoolXor ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::BOOL_XOR );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//BoolXor

BoolXor::~BoolXor () {}

void
BoolXor::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolXor::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolXor::consistency ()
{
}//consistency

//! It checks if
bool
BoolXor::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolXor::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



