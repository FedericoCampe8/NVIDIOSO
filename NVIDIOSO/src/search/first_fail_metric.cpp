//
//  first_fail_metric.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/11/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "first_fail_metric.h"

FirstFail::FirstFail () {
  _dbg = "FirstFailMetric - ";
  _metric_type = VariableChoiceMetricType::FIRST_FAIL;
}//InputOrder

int
FirstFail::compare ( double metric, Variable * var ) {
  double other_metric = (double) (var->domain_iterator)->domain_size ();
  if ( metric < other_metric ) {
    return 1;
  }
  else if ( metric == other_metric ) {
    return 0;
  }
  else {
    return -1;
  }
}//compare

int
FirstFail::compare ( Variable * var_a , Variable * var_b ) {
    return compare( (var_a->domain_iterator)->domain_size (), var_b );
}//compare

double
FirstFail::metric_value ( Variable * var ) {
  return (var->domain_iterator)->domain_size ();
}//metric_value

void
FirstFail::print () const {
  std::cout << "first_fail" << std::endl;
}//print



