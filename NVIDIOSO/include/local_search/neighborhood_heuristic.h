//
//  neighborhood_heuristic.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class represent a simple heuristic customizable by
//  the client with the given input parameters for selecting
//  next variable and next value for that variable.
//  More sophisticated should further specialize this simple heuristic
//  or specialize the base class heuristic.h.
// 

#ifndef __NVIDIOSO__neighborhood_heuristic__
#define __NVIDIOSO__neighborhood_heuristic__

#include "simple_heuristic.h"
#include "variable_choice_metric.h"
#include "value_choice_metric.h"

class NeighborhoodHeuristic;
typedef std::unique_ptr<NeighborhoodHeuristic> NeighborhoodHeuristicUPtr;
typedef std::shared_ptr<NeighborhoodHeuristic> NeighborhoodHeuristicSPtr;
 
class NeighborhoodHeuristic : public SimpleHeuristic {
public:
  	/**
  	 * Constructor, defines a new simple heuristic given
   	 * the metrics for selecting the next variable to label
   	 * and the value to assign to such variable.
   	 * @param vars a vector of pointer to variables to label.
   	 * @note if the variable metric is a nullptr, the next variable
   	 *       to label is the first non-ground variable (default).
   	 * @note default val_cm is indomain_min
   	 */
	NeighborhoodHeuristic ( std::vector< Variable* >& vars,
							VariableChoiceMetric * var_cm = nullptr,
                    		ValueChoiceMetric *    val_cm = nullptr );
  	
  	~NeighborhoodHeuristic ();
  	
  	/**
  	 * Get the variable which will be considered by this heuristic at next step.
  	 * @return index of the variable which will considered in the next iteration,
  	 *         or -1 if all variables have been already considered.
  	 */
  	int get_next_index () const;
  	
  	/**
  	 * Set current internal index.
  	 * @return previous internal index.
  	 */ 
  	int set_index ( int current_index );
  	
  	/**
	 * Change set of variables where local search is performed on.
	 * @note This function resets the internal state of the search.
	 */
	void set_search_variables ( std::vector< Variable* >& vars );
	
	void print () const;
};

#endif /* defined(__NVIDIOSO__neighborhood_heuristic__) */
