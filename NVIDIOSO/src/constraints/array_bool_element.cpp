//
//  array_bool_element.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_bool_element.h"

ArrayBoolElement::ArrayBoolElement () :
FZNConstraint ( ARRAY_BOOL_ELEMENT ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

ArrayBoolElement::ArrayBoolElement ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
ArrayBoolElement () {
  setup ( vars, args );
}//IntNe

ArrayBoolElement::~ArrayBoolElement () {}

void
ArrayBoolElement::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
ArrayBoolElement::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
ArrayBoolElement::consistency ()
{
}//consistency

//! It checks if
bool
ArrayBoolElement::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
ArrayBoolElement::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



