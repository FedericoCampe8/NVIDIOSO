//
//  indomain_min_metric.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "indomain_min_metric.h"

InDomainMin::InDomainMin () {
  _dbg = "InDomainMin - ";
  _metric_type = ValueChoiceMetricType::INDOMAIN_MIN;
}//InDomainMin

int
InDomainMin::metric_value ( Variable * var ) {
  return (var->domain_iterator)->min_val();
}//metric_value

void
InDomainMin::print () const {
  std::cout << "indomain_min" << std::endl;
}//print