//
//  variable_choice_metric.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface for the variable choice metrics
//  that can be used for defining a heuristic.
//  This interface is used to compare variables.
//

#ifndef __NVIDIOSO__variable_choice_metric__
#define __NVIDIOSO__variable_choice_metric__

#include "globals.h"
#include "variable.h"

enum class VariableChoiceMetricType {
  INPUT_ORDER,
  FIRST_FAIL,
  ANTI_FIRST_FAIL,
  SMALLEST,
  LARGEST,
  OCCURENCE,
  MOST_CONSTRAINED,
  MAX_REGRET,
  OTHER
};

class VariableChoiceMetric {
protected:
  //! Debug info
  std::string _dbg;
  
  VariableChoiceMetricType _metric_type;
  
  VariableChoiceMetric ();
  
public:
  virtual ~VariableChoiceMetric();
  
  /**
   * Get the type of metric for this variable choice metric.
   * @return the metric type of this variable choice metric.
   */
  virtual VariableChoiceMetricType metric_type () const;
  
  /**
   * Compares the metric value with a given variable.
   * @param metric the (metric) value to compare with.
   * @param var the (pointer to) variable to compare with the metric value.
   * @return  1 if metric is larger  than variable
   *          0 if metric is equal   to   variable
   *         -1 if metric is smaller than variable
   */
  virtual int compare ( double metric, Variable * var ) = 0;
  
  /**
   * Compares the metric value of var_a with the metric value of var_b.
   * @param var_a the (pointer to) variable to compare 
   *        with the metric value of var_b.
   * @param var_b the (pointer to) variable to compare 
   *        with the metric value of var_a.
   * @return  1 if var_a is larger  than var_b
   *          0 if var_a is equal   to   var_b
   *         -1 if var_a is smaller than var_b
   */
  virtual int compare ( Variable * var_a, Variable * var_b ) = 0;
  
  /**
   * Returns the value of the metric of a given variable.
   * @param var the variable for which the metric is required.
   * @return the value of the metric.
   */
  virtual double metric_value ( Variable * var ) = 0;
  
  //! Print info about this variable choice metric
  virtual void print () const = 0;
};

#endif /* defined(__NVIDIOSO__variable_choice_metric__) */
