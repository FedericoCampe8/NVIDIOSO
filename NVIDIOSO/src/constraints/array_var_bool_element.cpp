//
//  array_var_bool_element.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_var_bool_element.h"

ArrayVarBoolElement::ArrayVarBoolElement () :
FZNConstraint ( ARRAY_VAR_BOOL_ELEMENT ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

ArrayVarBoolElement::ArrayVarBoolElement ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
ArrayVarBoolElement () {
  setup ( vars, args );
}//IntNe

ArrayVarBoolElement::~ArrayVarBoolElement () {}

void
ArrayVarBoolElement::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
ArrayVarBoolElement::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
ArrayVarBoolElement::consistency ()
{
}//consistency

//! It checks if
bool
ArrayVarBoolElement::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
ArrayVarBoolElement::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



