//
//  objective_state.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/28/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class represents a simple container for the parameters defining the 
//  value of the objective function. 
//  This container is defined as a struct and it consists only on the header file.
//

#ifndef __NVIDIOSO__objective_state__
#define __NVIDIOSO__objective_state__

#include "globals.h"

struct ObjectiveState {

	//! Current number of unsatisfied constraints
	std::size_t number_unsat_constraint;
	
	//! Value of unsatisfiability correspondent to the current assignment
	double unsat_value;
	
	//! Value of the objective variable
	int obj_var_value;	
	
	// Indexes of the variables which have lead to the above values
	std::vector<int> neighborhood_index; 
	
	// Labelings of the variables which have lead to the above values
	std::vector<int> neighborhood_values;
	
	// Timestamp of when this object has been created
	std::time_t timestamp;
};  

#endif /* defined(__NVIDIOSO__objective_state__) */
