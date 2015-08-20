//
//  set_subset.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_subset.h"

SetSubset::SetSubset ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::SET_SUBSET );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetSubset

SetSubset::~SetSubset () {}

void
SetSubset::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetSubset::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetSubset::consistency ()
{
}//consistency

//! It checks if
bool
SetSubset::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetSubset::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



