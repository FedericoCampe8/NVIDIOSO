//
//  simple_search_memory_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a simple search memory manager.
//  A search memory manager is responsable for storing and implementing a finite 
//  set of memory states, which, in the case of SLS algorithms that do not use memory, 
//  may consist of a single state only, and in other cases holds information about 
//  the state of the search mechanism beyond the search position 
//  (e.g., tabu tenure values in the case of tabu search).
//  
//  @note the pattern used for the memory manager is similar to Memento pattern.
//

#ifndef __NVIDIOSO__simple_search_memory_manager__
#define __NVIDIOSO__simple_search_memory_manager__

#include "search_memory_manager.h"

class SimpleSearchMemoryManager : public SearchMemoryManager {
protected:

	/**
	 * Tenure of the memory manager represented as a hash table, where
	 * key: index of the variables in the neighborhood
	 * value: array of objective states
	 * @note the memory manager is in charge only of storing and retrieving the
	 *       objective state. In particular, it is not related to indexes of variables
	 *       or contents of objective states, which shall not be modified by 
	 *       the search memory manager.
	 */
	std::unordered_map< std::size_t, std::vector< ObjectiveState > > _tenure;
	
public:
	SimpleSearchMemoryManager ();
	
	~SimpleSearchMemoryManager ();
	
	//! Record a given state 
	void record_state ( ObjectiveState& state ) override;
	
	std::vector< ObjectiveState > get_state_from ( std::size_t idx, std::size_t k=1 ) override;
	
	/**
	 * Retrieve the states for the variable with index idx.
	 * @param idx index of the variable to query.
	 * @return a vector of objective states representing all the states 
	 *         stored in the tenure for the variable with index idx.
	 */
	std::vector< ObjectiveState > get_state ( std::size_t idx ) override;
	
	/**
	 * Retrieve the states for the variable with index idx.
	 * @param idx array of indexes of the variables to query.
	 * @return a vector of objective states representing all the states 
	 *         stored in the tenure for the variables with indexes in idx.
	 * @note this method does not distinguish between states that belong to different idx.
	 */ 
	std::vector< ObjectiveState > get_state ( std::vector< std::size_t > indexes ) override;
	
	/**
	 * Reset tenure only on a given variable.
	 * @param idx index of the variable to clear from the current tenure.
	 */
	void clear_memory ( std::size_t idx ) override;
	
	//! Reset tenure
	void clear_memory () override;
	
	//! Print info about the search memory manager and its current internal tenure
	void print () const override;
};

#endif /* defined(__NVIDIOSO__simple_search_memory_manager__) */
