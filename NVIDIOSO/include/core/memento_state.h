//
//  memento_state.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the internal state of a memento.
//

#ifndef __NVIDIOSO__memento_state__
#define __NVIDIOSO__memento_state__

#include "globals.h"

class MementoState {
public:
  virtual ~MementoState () {};
  
  //! Print information about this memento state
  virtual void print () const = 0;
};

#endif /* defined(__NVIDIOSO__memento_state__) */
