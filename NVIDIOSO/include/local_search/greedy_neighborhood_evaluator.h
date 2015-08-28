//
//  greedy_neighborhood_evaluator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/28/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a greedy evaluator function for neighborhood, i.e., 
//  it always returns the best labeling value found so far, best w.r.t. the 
//  value of the objective function. 
// 


#ifndef __NVIDIOSO__greedy_neighborhood_evaluator__
#define __NVIDIOSO__greedy_neighborhood_evaluator__

#include "neighborhood_evaluator.h"
 
class GreedyNeighborhoodEvaluator : public NeighborhoodEvaluator {
public:

	GreedyNeighborhoodEvaluator ();
  	
  	~GreedyNeighborhoodEvaluator ();
  	 
  	/**
  	 * Get best value for a given variable
  	 * according a greedy evaluator, i.e., the one associated with the best value.
  	 * @param obj_states an array of states to evaluate.
  	 * @return an objective state representing the best states 
  	 * found with a greedy algorithm.
  	 */
  	ObjectiveState get_best_value ( std::vector< ObjectiveState >& obj_states ) override;
	
	//! r-value version for get_best_value
	ObjectiveState get_best_value ( std::vector< ObjectiveState >&& obj_states ) override;
	
	//! Reset internal state of this evaluator
	void reset_evaluator () override;
	
	//! Print information about this evaluator
	void print () const override;
}; 

#endif /* defined(__NVIDIOSO__greedy_neighborhood_evaluator__) */
