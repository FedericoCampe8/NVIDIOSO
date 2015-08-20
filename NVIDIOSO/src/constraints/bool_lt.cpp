//
//  bool_lt.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_lt.h"

BoolLt::BoolLt ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::BOOL_LT );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//BoolLt

BoolLt::~BoolLt () {}

void
BoolLt::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolLt::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolLt::consistency ()
{
}//consistency

//! It checks if
bool
BoolLt::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolLt::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



