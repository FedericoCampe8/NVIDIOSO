//
//  simple_heuristic.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class represent a simple heuristic customizable by
//  the client with the given input parameters for selecting
//  next variable and next value for that variable.
//  More sophisticated should further specialize this simple heuristic
//  or specialize the base class heuristic.h.
//

#ifndef __NVIDIOSO__simple_heuristic__
#define __NVIDIOSO__simple_heuristic__

#include "heuristic.h"
#include "variable_choice_metric.h"
#include "value_choice_metric.h"

class SimpleHeuristic : public Heuristic {
protected:
  /**
   * The array of (pointers to) variables used
   * to store the references and hence to select
   * the next variable to label according to the
   * heuristic parameter specified as input.
   */
  std::vector< Variable* > _fd_variables;
  
  /**
   * The metric used to select the next variable
   * to label.
   */
  VariableChoiceMetric * _variable_metric;
  
  /**
   * The metric used to select the next value
   * to assign to the selected variable.
   */
  ValueChoiceMetric * _value_metric;
  
	/**
	 * It places the variable at index "variablePosition" at "searchPosition".
	 * @param searchPosition position currently considered by this heuristic
	 * @param variablePosition current position of the variable choosen by search. 
	 * @return variable choosen to be the choice point.
	 * @note this function actually swaps variables "searchPosition" with "variablePosition".
	 */
	void placeSearchVariable ( int searchPosition, int variablePosition );
  
public:
  /**
   * Constructor, defines a new simple heuristic given
   * the metrics for selecting the next variable to label
   * and the value to assign to such variable.
   * @param vars a vector of pointer to variables to label.
   * @param var_cm the variable metric used to select the next
   *        variable to label.
   * @param val_cm the value metric used to select the next
   *        value to assign to the selected variable.
   * @note if the variable metric is a nullptr, the next variable
   *       to label is the first non-ground variable.
   */
  SimpleHeuristic ( std::vector< Variable* > vars,
                    VariableChoiceMetric * var_cm,
                    ValueChoiceMetric *    val_cm );
  
  ~SimpleHeuristic ();
  
  /**
   * Gets next variable to label according to
   * the VariableChoiceMetric.
   * @param idx the index of the last variable
   *        returned by this heuristic.
   * @return a pointer to the next variable to label.
   */
  Variable * get_choice_variable ( int idx );
  
  /**
   * Returns the next value to assign
   * to the variable selected by this heuristic.
   * @return the value to assign to the selected variable.
   */
  int get_choice_value ();
  
  //! Print info about this heuristic
  void print () const;
};

#endif /* defined(__NVIDIOSO__simple_heuristic__) */
