//
//  array_var_int_element.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_var_int_element.h"

ArrayVarIntElement::ArrayVarIntElement () :
FZNConstraint ( ARRAY_VAR_INT_ELEMENT ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

ArrayVarIntElement::ArrayVarIntElement ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
ArrayVarIntElement () {
  setup ( vars, args );
}//IntNe

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
    FZNConstraint::print_semantic ();
}//print_semantic



