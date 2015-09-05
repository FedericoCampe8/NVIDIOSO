//
//  iterated_conditional_modes_heuristic.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements the "Iterated Conditional Modes" heuristic.
//

#ifndef __NVIDIOSO__iterated_conditional_modes_heuristic__
#define __NVIDIOSO__iterated_conditional_modes_heuristic__

#include "simple_local_search_heuristic.h"

class IteratedConditionalModesHeuristic : public SimpleLocalSearchHeuristic {
private:
 
	/**
	 * Vector of _fd_variables.size() size storing the current status
	 * of the search process. 
	 * Initially it is filled with the size of the domains for the 
	 * corresponding variables.
	 * A variable is completely assigned/visited if the value becomes zero.
	 * @note We assume here that the search process involves only ONE or ZERO
	 *       non ground variable, i.e., the size of the domains must be the same
	 *       during the whole search process and not decrease by constraint propagation.
	 */
	std::vector<int> _sampling_variable_status;

	//! Set heuristic for neighborhood exploration
	void set_neighborhood_heuristic ( VariableChoiceMetric * var_cm, ValueChoiceMetric * val_cm );
	
	//! Set neighborhood evaluator function
	void set_neighborhood_evaluator ( bool minimize, bool use_constraint_satisfiability_value, int percentage_on_sat );
	
protected:
	
	//! ICM starting neighborhood: first variable in _fd_variables (i.e., set {0}).
	std::vector<int> starting_neighborhood () override;
	
	/**
	 * Change status on _sampling_variable_status.
	 * @param var_index index of the variable in _sampling_variable_status.
	 */
	void notify_on_var_assignment ( int var_index ) override;
	
	/**
	 * Override neighborhood_last_assignment_on_var.
	 * For ICM heuristic, last assignment correspond to the last
	 * element of the domain of the given variable.
	 */
	bool neighborhood_last_assignment_on_var ( int var_index ) override;
	
	/**
	 * Override neighborhood_last_assignment_on_var.
	 * For ICM heuristic, complete assignment correspond to the one over the last
	 * element of the domain of the given variable.
	 */
	bool neighborhood_complete_assignment_on_var ( int var_index ) override;
	
public:
	
	/**
	 * Constructor.
	 * @param vars array of pointer to variable where the ICM is performed on.
	 * @param obj_var pointer to the objective variable.
	 * @param minimize Boolean value, true if obj value must be minimized, false otherwise.
	 * @param use_constraint_satisfiability_value Boolean value. True if the objective function
	 *        should consider the satisfiability of constraints instead of the number of 
	 *        satisfied constraints.
	 * @param percentage_on_sat percentage value (integer value in [0..100]) of the weight to assign
	 *        to the number of satisfied constraints/value of satisfiability w.r.t. the objective 
	 *        value assigned to the objective variable.
	 * @param var_cm pointer to a given variable choice metric. The client should rely on the 
	 *        default one.
	 * @param val_cm pointer to a given value choice metric. The client should rely on the 
	 *        default one.
	 */
	IteratedConditionalModesHeuristic ( std::vector< Variable* > vars,
										Variable * obj_var = nullptr,
										bool minimize = true,
										bool use_constraint_satisfiability_value = false, 
										int percentage_on_sat = 0,
										VariableChoiceMetric * var_cm = nullptr,
                    					ValueChoiceMetric *    val_cm = nullptr );
	
	virtual ~IteratedConditionalModesHeuristic ();
	
	//! Reset the internal state of the local search
	void reset_state () override;
	
	void print () const override;
};

#endif /* defined(__NVIDIOSO__iterated_conditional_modes_heuristic__) */
