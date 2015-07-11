//
//  anti_first_fail_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "anti_first_fail_metric.h"

AntiFirstFail::AntiFirstFail () {
    _dbg = "AntiFirstFailMetric - ";
    _metric_type = VariableChoiceMetricType::ANTI_FIRST_FAIL;
}//AntiFirstFail

int
AntiFirstFail::compare ( double metric, Variable * var )
{
    double other_metric = (double) (var->domain_iterator)->domain_size ();
    if ( metric < other_metric )
    {
        return -1;
    }
    else if ( metric == other_metric )
    {
        return 0;
    }
    else
    {
        return 1;
    }
}//compare

int
AntiFirstFail::compare ( Variable * var_a , Variable * var_b )
{
    return compare( (var_a->domain_iterator)->domain_size (), var_b );
}//compare

double
AntiFirstFail::metric_value ( Variable * var )
{
  return (var->domain_iterator)->domain_size ();
}//metric_value

void
AntiFirstFail::print () const
{
  std::cout << "anti_first_fail" << std::endl;
}//print



