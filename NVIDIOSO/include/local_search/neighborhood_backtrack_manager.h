//
//  neighborhood_backtrack_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a simple backtrack manager.
//  Backtrack is performed by storing the current state of the
//  objects represented by a Memento object.
//  Memento objects are store in a stack according to the current level.
//  Therefore, objects are restored according to a level given by
//  the client.
//

#ifndef __NVIDIOSO__neighborhood_backtrack_manager__
#define __NVIDIOSO__neighborhood_backtrack_manager__

#include "simple_backtrack_manager.h"

class NeighborhoodBacktrackManager;
typedef std::unique_ptr<NeighborhoodBacktrackManager> NeighborhoodBacktrackManagerUPtr; 
typedef std::shared_ptr<NeighborhoodBacktrackManager> NeighborhoodBacktrackManagerSPtr; 

class NeighborhoodBacktrackManager : public SimpleBacktrackManager {
protected:
	
	//! Current sequential id of the attached variables
	std::size_t _sequence_id;
	
	/**
   	 * Ordered list of backtrackable objects that are
   	 * subjects of this BacktrackManager observer.
   	 * Mapping between sequence ids and backtrackable objects ids
   	 */
  	std::unordered_map < std::size_t, std::size_t > _sequence_backtrackable_objects;
  
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
  	std::unordered_map < std::size_t, Memento * > _flat_trail_stack;
  
public:

  NeighborhoodBacktrackManager ();
  
  virtual ~NeighborhoodBacktrackManager();
  
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
   * Informs the manager that a given backtrackable object
   * has changed at a given level.
   * @param idx the (unique) id of the backtrackable object which is changed.
   * @note only object already registered with this manager
   *       can be restored later.
   */
  void add_changed ( size_t idx ) override;
  
  /**
   * Specifies the level which should become the
   * active one in the manager.
   * @param lvl the active level at which the changes will be recorded.
   */
  void set_level ( size_t lvl ) override;
  
  //! Reset the internal sequence id for backtracable objects
  virtual void reset_sequence_id ();
  
  /**
   * Removes a level.
   * It performs a backtrack from flat_stack on the variable with internal index id
   * @param id internal index (0, 1, 2, etc.) of the variable to restore from flat_stack.
   */
  virtual void remove_level_on_var ( std::size_t id ) ;
  
  
  //! Set the top of the stack as current flat stack
  virtual void flatten_stack ();
  
  //! Print information about this simple backtrack manager
  void print () const override;

};

#endif /* defined(__NVIDIOSO__neighborhood_backtrack_manager__) */
