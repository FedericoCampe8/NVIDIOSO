//
//  max_regret_metric.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  This class implements the "max_regret" variable choice metric.
//

#ifndef __NVIDIOSO__max_regret_metric__
#define __NVIDIOSO__max_regret_metric__

#include "variable_choice_metric.h"

class MaxRegret : public VariableChoiceMetric {
public:
    MaxRegret ();
    
    /**
     * Compare a metric value and a variable.
     * Metric is given by their largest difference between
     * the max and min values in its domain.
     */
    int compare ( double metric, Variable * var );
  
    /**
     * Compare variables w.r.t. their metrics.
     * Metric is given by their largest difference between
     * the max and min values in its domain.
     */
    int compare ( Variable * var_a, Variable * var_b );
  
    //! Get the metric value for first_fail
    double metric_value ( Variable * var );
  
    //! Print info
    void print () const;
};


#endif /* defined(__NVIDIOSO__first_fail_metric__) */
