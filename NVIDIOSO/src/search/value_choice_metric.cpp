//
//  value_choice_metric.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "value_choice_metric.h"

ValueChoiceMetric::ValueChoiceMetric () :
_dbg         ( "ValueChoiceMetric - "),
_metric_type ( ValueChoiceMetricType::OTHER ) {
}//ValueChoiceMetric

ValueChoiceMetric::~ValueChoiceMetric () {
}//ValueChoiceMetric

ValueChoiceMetricType
ValueChoiceMetric::metric_type () const {
  return _metric_type;
}//metric_type

int 
ValueChoiceMetric::metric_value ( Variable * var, int comparator )
{
	return metric_value ( var );
}//metric_value