//
//  array_var_int_element.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_var_int_element.h"

ArrayVarIntElement::ArrayVarIntElement ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::ARRAY_VAR_INT_ELEMENT );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//ArrayVarIntElement

ArrayVarIntElement::~ArrayVarIntElement () {}

void
ArrayVarIntElement::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
ArrayVarIntElement::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
ArrayVarIntElement::consistency ()
{
}//consistency

//! It checks if
bool
ArrayVarIntElement::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
ArrayVarIntElement::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



