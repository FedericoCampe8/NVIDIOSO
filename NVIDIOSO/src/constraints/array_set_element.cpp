//
//  array_set_element.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_set_element.h"

ArraySetElement::ArraySetElement () :
FZNConstraint ( ARRAY_SET_ELEMENT ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

ArraySetElement::ArraySetElement ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
ArraySetElement () {
  setup ( vars, args );
}//IntNe

ArraySetElement::~ArraySetElement () {}

void
ArraySetElement::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
ArraySetElement::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
ArraySetElement::consistency ()
{
}//consistency

//! It checks if
bool
ArraySetElement::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
ArraySetElement::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



