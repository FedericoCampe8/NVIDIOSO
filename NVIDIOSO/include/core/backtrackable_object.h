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


class BacktrackableObject {
protected:
  //! Unique identifier for this backtrackable object.
  int _backtrackable_id;
  
  //! Memento hold by this this backtrackable object.
  MementoState * _current_state;
  
public:
  /**
   * Create a new memento object (state).
   * @return a reference to a new memento.
   */
  virtual Memento * create_memento () {
    Memento * m = new Memento ();
    m->set_state( _current_state );
    return m;
  }//create_memento
  
  /**
   * Set a memento as current state.
   * @param m the memento to set as current state.
   */
  virtual void set_memento ( Memento& m ) {
    _current_state = m.get_state();
  }//set_memento
  
  /**
   * Set the current state of this backtrackable object.
   * @param state the current state to set.
   */
  virtual void set_state ( MementoState * state ) {
    _current_state = state;
  }//set_state
  
  /**
   * Returns the unique id of this backtrackable object.
   * @return the unique id of this backtrackable object.
   */
  virtual int get_backtrackable_id () const {
    return _backtrackable_id;
  }//get_id
  
  /**
   * Set unique id for this backtrackable object.
   * Concrete backtracable objects are required to
   * implement this method so any backtrackable object
   * has its unique id.
   */
  virtual void set_backtrackable_id () = 0;
  
  /**
   * Restore a state from the current state
   * hold by the BacktrackableObject.
   */
  virtual void restore_state () = 0;
  
  /**
   * Set internal state with other information
   * hold by concrete BacktrackableObject objects.
   */
  virtual void set_state () = 0;
};



#endif /* defined(__NVIDIOSO__backtrackable_object__) */
