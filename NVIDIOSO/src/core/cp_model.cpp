//
//  cp_model.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cp_model.h"

CPModel::CPModel () {
}//CPModel

CPModel::~CPModel () {
}//~CPModel

void
CPModel::add_variable ( VariablePtr vpt ) {
  assert( vpt != nullptr );
  vpt->print ();
}//add_variable

void
CPModel::add_constraint ( ConstraintPtr ) {
}//add_constraint

void
CPModel::add_search_engine ( SearchEnginePtr ) {
}//add_search_engine




