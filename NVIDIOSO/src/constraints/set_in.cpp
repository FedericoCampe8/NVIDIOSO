//
//  set_in.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_in.h"

SetIn::SetIn ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::SET_IN );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetIn

SetIn::~SetIn () {}

void
SetIn::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetIn::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetIn::consistency ()
{
}//consistency

//! It checks if
bool
SetIn::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetIn::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



