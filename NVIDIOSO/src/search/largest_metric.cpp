//
//  largest_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "largest_metric.h"

Largest::Largest () {
    _dbg = "LargestMetric - ";
    _metric_type = VariableChoiceMetricType::LARGEST;
}//Largest

int
Largest::compare ( double metric, Variable * var )
{
    double other_metric = (double) (var->domain_iterator)->max_val ();
    if ( metric > other_metric )
    {
        return 1;
    }
    else if ( metric == other_metric )
    {
        return 0;
    }
    else
    {
        return -1;
    }
}//compare

int
Largest::compare ( Variable * var_a , Variable * var_b )
{
    return compare( (var_a->domain_iterator)->max_val (), var_b );
}//compare

double
Largest::metric_value ( Variable * var )
{
  return (var->domain_iterator)->max_val ();
}//metric_value

void
Largest::print () const
{
  std::cout << "largest" << std::endl;
}//print



