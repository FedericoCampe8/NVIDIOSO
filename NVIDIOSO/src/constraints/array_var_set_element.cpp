//
//  array_var_set_element.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_var_set_element.h"

ArrayVarSetElement::ArrayVarSetElement ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::ARRAY_VAR_SET_ELEMENT );
  /* 
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//ArrayVarSetElement

ArrayVarSetElement::~ArrayVarSetElement () {}

void
ArrayVarSetElement::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
ArrayVarSetElement::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
ArrayVarSetElement::consistency ()
{
}//consistency

//! It checks if
bool
ArrayVarSetElement::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
ArrayVarSetElement::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



