//
//  indomain_max_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/11/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "indomain_max_metric.h"

InDomainMax::InDomainMax () {
  _dbg = "InDomainMax - ";
  _metric_type = ValueChoiceMetricType::INDOMAIN_MAX;
}//InDomainMin

int
InDomainMax::metric_value ( Variable * var ) 
{
  return (var->domain_iterator)->max_val ();
}//metric_value

void
InDomainMax::print () const {
  std::cout << "indomain_max" << std::endl;
}//print
