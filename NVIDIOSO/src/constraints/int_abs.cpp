//
//  int_abs.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_abs.h"

IntAbs::IntAbs () :
FZNConstraint ( INT_ABS ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

IntAbs::IntAbs ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntAbs () {
  setup ( vars, args );
}//IntNe

IntAbs::~IntAbs () {}

void
IntAbs::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntAbs::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntAbs::consistency ()
{
}//consistency

//! It checks if
bool
IntAbs::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntAbs::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



