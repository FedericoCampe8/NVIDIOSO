//
//  search_memory_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class declares the interface for a search memory manager.
//  A search memory manager is responsable for storing and implementing a finite 
//  set of memory states, which, in the case of SLS algorithms that do not use memory, 
//  may consist of a single state only, and in other cases holds information about 
//  the state of the search mechanism beyond the search position 
//  (e.g., tabu tenure values in the case of tabu search).
//
//  @note the pattern used for the memory manager is similar to Memento pattern.
//

#ifndef __NVIDIOSO__search_memory_manager__
#define __NVIDIOSO__search_memory_manager__

#include "globals.h"
#include "objective_state.h"

class SearchMemoryManager;
typedef std::unique_ptr<SearchMemoryManager> MemoryManagerUPtr;
typedef std::shared_ptr<SearchMemoryManager> MemoryManagerSPtr; 

class SearchMemoryManager {
public:
	virtual ~SearchMemoryManager () {}
	
	/**
	 * Record an objective state.
	 * @param state instance of objective state to store into the internal tenure.
	 */
	virtual void record_state ( ObjectiveState& state ) = 0;
	 
	/**
	 * Retrieve the most recent k states for the variable with index idx.
	 * @param idx index of the variable to query.
	 * @param k index (from the end) of the states to retrieve (default, last one).
	 * @return a vector of objective states representing the most recent k states 
	 *         stored in the tenure for the variable with index idx.
	 */
	virtual std::vector< ObjectiveState > get_state_from ( std::size_t idx, std::size_t k=1 ) = 0;
	
	/**
	 * Retrieve the states for the variable with index idx.
	 * @param idx index of the variable to query.
	 * @return a vector of objective states representing all the states 
	 *         stored in the tenure for the variable with index idx.
	 */
	virtual std::vector< ObjectiveState > get_state ( std::size_t idx ) = 0;
	  
	/**
	 * Retrieve the states for the variable with index idx.
	 * @param indexes array of indexes of the variables to query.
	 * @return a vector of objective states representing all the states 
	 *         stored in the tenure for the variables with indexes in idx.
	 * @note this method does not distinguish between states that belong to different idx.
	 */
	virtual std::vector< ObjectiveState > get_state ( std::vector< std::size_t > indexes ) = 0;
	
	/**
	 * Reset tenure only on a given variable.
	 * @param idx index of the variable to clear from the current tenure.
	 */
	virtual void clear_memory ( std::size_t idx ) = 0;
	
	//! Reset tenure 
	virtual void clear_memory () = 0;
	
	//! Print info about the search memory manager and its current internal tenure
	virtual void print () const = 0;
};   

#endif /* defined(__NVIDIOSO__search_memory_manager__) */
