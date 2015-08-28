//
//  neighborhood_evaluator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/24/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class represent the evaluator function for neighborhood, i.e., 
//  it is the interface for the class implementing the objective function.
//  Instances of this class are used to determine the best labels
//  for the variables in the neighborhood explored by a local search strategy.
// 


#ifndef __NVIDIOSO__neighborhood_evaluator__
#define __NVIDIOSO__neighborhood_evaluator__

#include "globals.h"
#include "variable.h"
#include "search_memory_manager.h"

class NeighborhoodEvaluator;
typedef std::unique_ptr<NeighborhoodEvaluator> NeighborhoodEvaluatorUPtr;
typedef std::shared_ptr<NeighborhoodEvaluator> NeighborhoodEvaluatorSPtr;
 
class NeighborhoodEvaluator {
public: 
  	virtual ~NeighborhoodEvaluator () {};
  	
  	/**
  	 * Get best value for a given variable
  	 * according to the objective value, the set of states given in input,
  	 * and the internal state of the evaluator.
  	 * @param obj_states an array of states to evaluate.
  	 * @return an objective state representing the best one according to the evaluator.
  	 */
  	virtual ObjectiveState get_best_value ( std::vector< ObjectiveState >& obj_states )  = 0;
	
	//! r-value version for get_best_value
	virtual ObjectiveState get_best_value ( std::vector< ObjectiveState >&& obj_states ) = 0;
	
	//! Reset internal evaluator's state
	virtual void reset_evaluator () = 0;
	
	//! Print information about the evaluator
	virtual void print () const = 0;
};

#endif /* defined(__NVIDIOSO__neighborhood_evaluator__) */
