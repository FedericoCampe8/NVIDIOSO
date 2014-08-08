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

class BacktrackManager {
protected:
  //! Debug info
  std::string _dbg;
  
  //! Current active level in the manager
  size_t _current_level;
  
  BacktrackManager ();
  
public:
  virtual ~BacktrackManager ();
  
  /**
   * Get the current active level.
   * @return current active level in the manager.
   */
  virtual size_t get_level () const;
  
  /**
   * Specifies the level which should become the
   * active one in the manager.
   * @param lvl the active level at which the changes will be recorded.
   */
  virtual void set_level ( size_t lvl ) = 0;
  
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
  
  //! Print info about the manager
  virtual void print () = 0;
};


#endif /* defined(__NVIDIOSO__backtrack_manager__) */
