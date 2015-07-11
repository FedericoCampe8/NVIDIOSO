//
//  most_constrained_metric.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  This class implements the "most_constrained" variable choice metric.
//

#ifndef __NVIDIOSO__most_constrained_metric__
#define __NVIDIOSO__most_constrained_metric__

#include "variable_choice_metric.h"

class MostConstrained : public VariableChoiceMetric {
public:
    MostConstrained ();
    
    /**
     * Compare a metric value and a variable.
     * Metric is given by their smallest domains,
     * breaking ties using the number of constraints.
     */
    int compare ( double metric, Variable * var );
  
    /**
     * Compare variables w.r.t. their metrics.
     * Metric is given by their smallest domains,
     * breaking ties using the number of constraints.
     */
    int compare ( Variable * var_a, Variable * var_b );
  
    //! Get the metric value for first_fail
    double metric_value ( Variable * var );
  
    //! Print info
    void print () const;
};


#endif /* defined(__NVIDIOSO__first_fail_metric__) */
