//
//  memento.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 06/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Class for a memento objects.
//  A memento object is a backtrackable object that is used to store the
//  current state of another object called "originator" or
//  "backtrackable" object.
//  The backtrackable manager or Caretaker uses mementos to restore a
//  state of an originator.
//  A class that implements this interface is responsable to define
//  the meaning of "state" of a given originator.
//

#ifndef __NVIDIOSO__memento__
#define __NVIDIOSO__memento__

#include "globals.h"
#include "memento_state.h"

class BacktrackableObject;

class Memento {
protected:
  // Private members accessible only to BacktrackableObjects
  friend class BacktrackableObject;
  
  // State representing the memento
  MementoState * _memento_state;
  
  /**
   * Set a state as a memento object.
   * @param state the current state representing a mememnto object.
   */
  virtual void set_state ( MementoState * state ) {
    _memento_state = state;
  }//set_state
  
  /**
   * Get the current state saved as memento.
   * @return the current state/memento.
   */
  virtual MementoState * get_state () {
    return _memento_state;
  }//get_state
  
  //! Protected constructor
  Memento() {};
  
public:
  // Narrow public interface
  virtual ~Memento () {};
  
};


#endif /* defined(__NVIDIOSO__memento__) */
