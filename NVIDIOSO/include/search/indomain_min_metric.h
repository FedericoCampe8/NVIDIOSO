//
//  indomain_min_metric.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements the indomain_min metric for value choice.
//

#ifndef __NVIDIOSO__indomain_min_metric__
#define __NVIDIOSO__indomain_min_metric__

#include "value_choice_metric.h"

class InDomainMin : public ValueChoiceMetric {
public:
  InDomainMin ();
  
  /**
   * Gets value to assign to var using
   * indomain_min choice.
   * @param var the (pointer to) variable for which a value if needed.
   * @return the value to assign to var.
   */
  int metric_value ( Variable * var );
  
  void print () const;
};

#endif /* defined(__NVIDIOSO__indomain_min_metric__) */
