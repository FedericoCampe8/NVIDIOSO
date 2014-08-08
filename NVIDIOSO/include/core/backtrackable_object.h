//
//  backtrackable_object.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 06/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents an interface "Originator" for memento object.
//  It defines how a memento object must be represented (i.e., its state).
//  This representation is then used to restore states when performing
//  backtrack during search.
//


#ifndef __NVIDIOSO__backtrackable_object__
#define __NVIDIOSO__backtrackable_object__

#include "globals.h"
#include "memento.h"

template <class T>
class BacktrackableObject {
protected:
  T _current_state;
  
public:
  /**
   * Create a new memento object (state).
   * @return a reference to a new memento.
   */
  virtual Memento<T> * create_memento () {
    Memento<T> * m = new Memento<T> ();
    m->set_state( _current_state );
    return m;
  }//create_memento
  
  /**
   * Set a memento a current state.
   * @param m the memento to set as current state.
   */
  virtual void set_memento ( Memento<T>& m ) {
    _current_state = m.get_state();
  }//set_memento

  /**
   * Set the current state of this BacktrackableObject.
   * @param state the current state to set.
   */
  virtual void set_state ( T state ) {
    _current_state = state;
  }//set_state

};



#endif /* defined(__NVIDIOSO__backtrackable_object__) */
