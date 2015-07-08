//
//  array_bool_or.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_bool_or.h"

ArrayBoolOr::ArrayBoolOr () :
FZNConstraint ( ARRAY_BOOL_OR ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

ArrayBoolOr::ArrayBoolOr ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
ArrayBoolOr () {
  setup ( vars, args );
}//IntNe

ArrayBoolOr::~ArrayBoolOr () {}

void
ArrayBoolOr::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
ArrayBoolOr::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
ArrayBoolOr::consistency ()
{
}//consistency

//! It checks if
bool
ArrayBoolOr::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
ArrayBoolOr::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



