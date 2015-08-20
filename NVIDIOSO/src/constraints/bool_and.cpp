//
//  bool_and.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_and.h"

BoolAnd::BoolAnd ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::BOOL_AND );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//BoolAnd

BoolAnd::~BoolAnd () {}

void
BoolAnd::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolAnd::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolAnd::consistency ()
{
}//consistency

//! It checks if
bool
BoolAnd::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolAnd::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



