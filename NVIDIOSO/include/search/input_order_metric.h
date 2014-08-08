//
//  input_order_metric.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements the "input_order" variable choice metric.
//

#ifndef __NVIDIOSO__input_order_metric__
#define __NVIDIOSO__input_order_metric__

#include "variable_choice_metric.h"

class InputOrder : public VariableChoiceMetric {
public:
  InputOrder ();
  
  /**
   * Compare a metric value and a variable.
   * Metric is given by the id of the vars as they have been
   * defined when instantiated.
   */
  int compare ( double metric, Variable * var );
  
  /**
   * Compare variables w.r.t. their metrics.
   * Metric is given by the id of the vars as they have been
   * defined when instantiated.
   */
  int compare ( Variable * var_a, Variable * var_b );
  
  //! Get the metric value for input_order
  double metric_value ( Variable * var );
  
  //! Print info
  void print () const;
};

#endif /* defined(__NVIDIOSO__input_order_metric__) */
