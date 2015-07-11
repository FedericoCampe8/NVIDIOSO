//
//  occurence_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "occurence_metric.h"

Occurence::Occurence () {
    _dbg = "OccurenceMetric - ";
    _metric_type = VariableChoiceMetricType::OCCURENCE;
}//Occurence

int
Occurence::compare ( double metric, Variable * var )
{
    double other_metric = (double) var->size_constraints_original ();
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
Occurence::compare ( Variable * var_a , Variable * var_b )
{
    return compare( var_a->size_constraints_original (), var_b );
}//compare

double
Occurence::metric_value ( Variable * var )
{
  return var->size_constraints_original ();
}//metric_value

void
Occurence::print () const
{
  std::cout << "occurence" << std::endl;
}//print



