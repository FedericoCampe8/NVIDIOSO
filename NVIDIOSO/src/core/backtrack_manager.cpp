//
//  backtrack_manager.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "backtrack_manager.h"

BacktrackManager::BacktrackManager () :
_dbg           ( "BacktrackManager - " ),
_current_level ( 0 ) {
}//BacktrackManager

BacktrackManager::~BacktrackManager () {
}//~BacktrackManager

size_t
BacktrackManager::get_level () const {
  return _current_level;
}//get_level

