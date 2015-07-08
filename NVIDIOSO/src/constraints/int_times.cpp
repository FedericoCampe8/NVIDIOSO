//
//  int_times.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_times.h"

IntTimes::IntTimes () :
FZNConstraint ( INT_TIMES ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntTimes

IntTimes::IntTimes ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntTimes () {
  setup ( vars, args );
}//IntTimes

IntTimes::~IntTimes () {}

void
IntTimes::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntTimes::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntTimes::consistency ()
{
}//consistency

//! It checks if
bool
IntTimes::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntTimes::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



