//
//  set_union.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_union.h"

SetUnion::SetUnion ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::SET_UNION );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetUnion

SetUnion::~SetUnion () {}

void
SetUnion::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetUnion::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetUnion::consistency ()
{
}//consistency

//! It checks if
bool
SetUnion::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetUnion::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



