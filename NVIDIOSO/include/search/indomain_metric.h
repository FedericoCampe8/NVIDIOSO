//
//  indomain_metric.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  This class implements the indomain_min metric for value choice.
//

#ifndef __NVIDIOSO__indomain_metric__
#define __NVIDIOSO__indomain_metric__

#include "value_choice_metric.h"

class InDomain : public ValueChoiceMetric {
public:
  InDomain ();
  
  /**
   * Gets value to assign to var using indomain choice.
   * @param var the (pointer to) variable for which a value if needed.
   * @return the value to assign to var.
   */
  int metric_value ( Variable * var );
  
  void print () const;
};

#endif 
