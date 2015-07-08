//
//  set_subset.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_subset.h"

SetSubset::SetSubset () :
FZNConstraint ( SET_SUBSET ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetSubset

SetSubset::SetSubset ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetSubset () {
  setup ( vars, args );
}//SetSubset

SetSubset::~SetSubset () {}

void
SetSubset::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetSubset::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetSubset::consistency ()
{
}//consistency

//! It checks if
bool
SetSubset::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetSubset::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



