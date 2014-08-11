//
//  simple_backtrack_manager.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements a simple backtrack manager.
//  Backtrack is performed by storing the current state of the
//  objects represented by a Memento object.
//  Memento objects are store in a stack according to the current level.
//  Therefore, objects are restored according to a level given by
//  the client.
//

#ifndef __NVIDIOSO__simple_backtrack_manager__
#define __NVIDIOSO__simple_backtrack_manager__

#include "backtrack_manager.h"
#include "memento.h"

class SimpleBacktrackManager : public BacktrackManager {
protected:
  //! Debug info
  std::string _dbg;
  
  //! Current active level in the manager
  size_t _current_level;
  
  /**
   * Ordered list of backtrackable objects that are
   * subjects of this BacktrackManager observer.
   */
  std::map < size_t, BacktrackableObject * > _backtrackable_objects;
  
  /**
   * Set of changed backtrackable objects.
   * When the set_level method is called, the objects in 
   * this list will be considered for saving their memento objects
   * (i.e., their state).
   */
  std::set < size_t > _changed_backtrackables;
  
  /**
   * Stack of list of Mementos to restore when the method
   * remove_level is invoked. The states of the backtrackable
   * objects will be re-stored from here.
   * Each object in the trail stack is a pair where the first element
   * represents the level in which the second element 
   * (pairs of backtrackable object and memento objects) are stored.
   * For example, at a given level:
   * < level, [ (id_1, Memento_1), (id_2, Memento_2), ... ] >
   */
  std::stack< std::pair < size_t,
  std::vector< std::pair < size_t, Memento * > > > > _trail_stack;
  
  /**
   * Stack used to store auxiliary information for each
   * level of the trail stack. Using this stack the
   * backtrack process can be speeded up re-setting only 
   * the most memento of each backtrackable object.
   * @todo implement this functionality.
   */
  std::stack < std::vector< size_t > > _trail_stack_info;
  
public:
  SimpleBacktrackManager ();
  virtual ~SimpleBacktrackManager();
  
  /**
   * Register a backtrackable object to this manager using the unique id
   * of the  backtrackable object.
   * @param bkt_obj a reference to a backtrackable object.
   */
  void attach_backtracable ( BacktrackableObject * bkt_obj );
  
  /**
   * Detaches a backtrackable object fromt this manager, so
   * its state won't be restored anymore.
   * @param bkt_id the id of the backtrackable object to detach.
   */
  void detach_backtracable ( size_t bkt_id );
  
  /**
   * Get the current active level.
   * @return current active level in the manager.
   */
  size_t get_level () const;
  
  /**
   * Informs the manager that a given backtrackable object
   * has changed at a given level.
   * @param idx the (unique) id of the backtrackable object which is changed.
   * @note only object already registered with this manager
   *       can be restored later.
   */
  void add_changed ( size_t idx );
  
  /**
   * Specifies the level which should become the
   * active one in the manager.
   * @param lvl the active level at which the changes will be recorded.
   */
  void set_level ( size_t lvl ) override;
  
  /**
   * Forces the storage of all the backtrackable objects
   * attached to this manager (at next set_level call),
   * no matter if a backtrackable object has been modified or not.
   */
  void force_storage () override;
  
  /**
   * Removes a level.
   * It performs a backtrack from that level.
   * @param lvl the level which is being removed.
   */
  void remove_level ( size_t lvl ) override;
  
  /**
   * Removes all levels until the one given as input.
   * It performs backtrack until the level given as input.
   * @param lvl the level to backtrack to.
   */
  void remove_until_level ( size_t lvl ) override;
  
  /**
   * Returns the number of backtrackable objects attached
   * to this backtrack manager.
   * @return number of objects attached to this manager.
   */
  size_t number_backtracable () const override;
  
  /**
   * Returns the number of changed backtrackable objects
   * from last call to set_level in this backtrack manager.
   * @return number of changed objects.
   */
  size_t number_changed_backtracable () const override;
  
  //! Print information about this simple backtrack manager
  void print () const override;

};

#endif /* defined(__NVIDIOSO__simple_backtrack_manager__) */
