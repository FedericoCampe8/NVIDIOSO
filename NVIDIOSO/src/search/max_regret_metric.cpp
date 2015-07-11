//
//  max_regret_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "max_regret_metric.h"

MaxRegret::MaxRegret () {
    _dbg = "MaxRegret - ";
    _metric_type = VariableChoiceMetricType::MAX_REGRET;
}//MaxRegret

int
MaxRegret::compare ( double metric, Variable * var )
{
    double other_metric = (double) ((var->domain_iterator)->max_val ()) - ((var->domain_iterator)->min_val ());
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
MaxRegret::compare ( Variable * var_a , Variable * var_b )
{
    return compare( ((var_a->domain_iterator)->max_val ()) - ((var_a->domain_iterator)->min_val ()), var_b );
}//compare

double
MaxRegret::metric_value ( Variable * var )
{
    return  ((var->domain_iterator)->max_val ()) - ((var->domain_iterator)->min_val ());
}//metric_value

void
MaxRegret::print () const
{
  std::cout << "max_regret" << std::endl;
}//print



