//
//	array_bool_and.cpp  
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "array_bool_and.h"

ArrayBoolAnd::ArrayBoolAnd ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::ARRAY_BOOL_AND );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//ArrayBoolAnd

ArrayBoolAnd::~ArrayBoolAnd () {}

void
ArrayBoolAnd::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
ArrayBoolAnd::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
ArrayBoolAnd::consistency ()
{
}//consistency

//! It checks if
bool
ArrayBoolAnd::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
ArrayBoolAnd::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



