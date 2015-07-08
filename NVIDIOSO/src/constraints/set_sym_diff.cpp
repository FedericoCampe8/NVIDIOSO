//
//  set_sym_diff.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_sym_diff.h"

SetSymDiff::SetSymDiff () :
FZNConstraint ( SET_SYMDIFF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetSymDiff

SetSymDiff::SetSymDiff ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetSymDiff () {
  setup ( vars, args );
}//SetSymDiff

SetSymDiff::~SetSymDiff () {}

void
SetSymDiff::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetSymDiff::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetSymDiff::consistency ()
{
}//consistency

//! It checks if
bool
SetSymDiff::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetSymDiff::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



