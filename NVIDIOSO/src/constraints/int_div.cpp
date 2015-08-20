//
//  int_div.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_div.h"

IntDiv::IntDiv ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::INT_DIV );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntDiv

IntDiv::~IntDiv () {}

void
IntDiv::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntDiv::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntDiv::consistency ()
{
}//consistency

//! It checks if
bool
IntDiv::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntDiv::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



