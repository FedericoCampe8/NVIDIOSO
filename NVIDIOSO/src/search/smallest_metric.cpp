//
//  smallest_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "smallest_metric.h"

Smallest::Smallest () {
    _dbg = "SmallestMetric - ";
    _metric_type = VariableChoiceMetricType::SMALLEST;
}//Smallest

int
Smallest::compare ( double metric, Variable * var )
{
    double other_metric = (double) (var->domain_iterator)->min_val ();
    if ( metric < other_metric )
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
Smallest::compare ( Variable * var_a , Variable * var_b )
{
    return compare( (var_a->domain_iterator)->min_val (), var_b );
}//compare

double
Smallest::metric_value ( Variable * var )
{
  return (var->domain_iterator)->min_val ();
}//metric_value

void
Smallest::print () const
{
  std::cout << "smallest" << std::endl;
}//print



