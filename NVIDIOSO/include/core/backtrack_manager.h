//
//  backtrack_manager.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class representes the interface for a Backtrack manager, i.e., the
//  "Caretaker" who is in charge of managing the list of Memento objects.
//  This is the interface for the client code to access.
//  It stores Memento objects and re-set a previous state effectively
//  performing backtrack.
//

#ifndef __NVIDIOSO__backtrack_manager__
#define __NVIDIOSO__backtrack_manager__

#include "globals.h"
#include "backtrackable_object.h"

class BacktrackManager;
typedef std::shared_ptr<BacktrackManager> BacktrackManagerPtr;

class BacktrackManager {
public:
  virtual ~BacktrackManager () {};
  
  /**
   * Register a backtrackable object to this manager using the unique id
   * of the  backtrackable object.
   * @param bkt_obj a reference to a backtrackable object.
   */
  virtual void attach_backtracable ( BacktrackableObject * bkt_obj ) = 0;
  
  /**
   * Detaches a backtrackable object fromt this manager, so
   * its state won't be restored anymore.
   * @param bkt_id the id of the backtrackable object to detach.
   */
  virtual void detach_backtracable ( size_t bkt_id ) = 0;
  
  /**
   * Informs the manager that a given backtrackable object
   * has changed at a given level.
   * @param idx the (unique) id of the backtrackable object which is changed.
   */
  virtual void add_changed ( size_t idx ) = 0;
  
  /**
   * Get the current active level.
   * @return current active level in the manager.
   */
  virtual size_t get_level () const = 0;
  
  /**
   * Specifies the level which should become the
   * active one in the manager.
   * @param lvl the active level at which the changes will be recorded.
   */
  virtual void set_level ( size_t lvl ) = 0;
  
  /**
   * Forces the storage of all the backtrackable objects
   * attached to this manager (at next set_level call),
   * no matter if a backtrackable object has been modified or not.
   */
  virtual void force_storage () = 0;
  
  /**
   * Removes a level. 
   * It performs a backtrack from that level.
   * @param lvl the level which is being removed.
   */
  virtual void remove_level ( size_t lvl ) = 0;
  
  /**
   * Removes all levels until the one given as input.
   * It performs backtrack until the level given as input.
   * @param lvl the level to backtrack to.
   */
  virtual void remove_until_level ( size_t lvl ) = 0;
  
  /**
   * Returns the number of backtrackable objects attached
   * to this backtrack manager.
   * @return number of objects attached to this manager.
   */
  virtual size_t number_backtracable () const = 0;
  
  /**
   * Returns the number of changed backtrackable objects
   * from last call to set_level in this backtrack manager.
   * @return number of changed objects.
   */
  virtual size_t number_changed_backtracable () const = 0;
  
  //! Print information about this backtrack manager
  virtual void print () const = 0;
};


#endif /* defined(__NVIDIOSO__backtrack_manager__) */
