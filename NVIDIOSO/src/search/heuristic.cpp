//
//  heuristic.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "heuristic.h"

Heuristic::Heuristic () :
_dbg           ( "Heuristic - " ),
_current_index ( 0 ) {
}//Heuristic

Heuristic::~Heuristic () {
}//~Heuristic

int
Heuristic::get_index () const {
  return _current_index;
}//get_index
