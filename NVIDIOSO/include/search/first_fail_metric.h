//
//  first_fail_metric.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 11/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements the "first_fail" variable choice metric.
//

#ifndef __NVIDIOSO__first_fail_metric__
#define __NVIDIOSO__first_fail_metric__

#include "variable_choice_metric.h"

class FirstFail : public VariableChoiceMetric {
public:
  FirstFail ();
  
  /**
   * Compare a metric value and a variable.
   * Metric is given by their domain's size.
   */
  int compare ( double metric, Variable * var );
  
  /**
   * Compare variables w.r.t. their metrics.
   * Metric is given by their domain's size.
   */
  int compare ( Variable * var_a, Variable * var_b );
  
  //! Get the metric value for first_fail
  double metric_value ( Variable * var );
  
  //! Print info
  void print () const;
};


#endif /* defined(__NVIDIOSO__first_fail_metric__) */
