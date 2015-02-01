//
//  variable_choice_metric.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "variable_choice_metric.h"

VariableChoiceMetric::VariableChoiceMetric() :
_dbg         ("VariableChoiceMetric - "),
_metric_type ( VariableChoiceMetricType::OTHER ) {
}//VariableChoiceMetric

VariableChoiceMetric::~VariableChoiceMetric() {
}//~VariableChoiceMetric

VariableChoiceMetricType
VariableChoiceMetric::metric_type () const {
  return _metric_type;
}//get_metric

