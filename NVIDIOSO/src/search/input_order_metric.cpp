//
//  input_order_metric.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "input_order_metric.h"

InputOrder::InputOrder () {
  _dbg = "InputOrderMetric - ";
  _metric_type = VariableChoiceMetricType::INPUT_ORDER;
}//InputOrder

int
InputOrder::compare ( double metric, Variable * var ) {
  int other_metric = var->get_id ();
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
InputOrder::compare ( Variable * var_a , Variable * var_b ) {
  return compare( var_a->get_id(), var_b );
}//compare

double
InputOrder::metric_value ( Variable * var ) {
  return var->get_id();
}//metric_value

void
InputOrder::print () const {
  std::cout << "input_order" << std::endl;
}//print


