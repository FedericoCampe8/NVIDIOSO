//
//  most_constrained__metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <math.h>
#include "most_constrained_metric.h"

MostConstrained::MostConstrained () {
    _dbg = "MostConstrainedMetric - ";
    _metric_type = VariableChoiceMetricType::MOST_CONSTRAINED;
}//MostConstrained

int
MostConstrained::compare ( double metric, Variable * var )
{
    double other_metric = (double) (var->domain_iterator)->domain_size ();

    // Get the domain size and number of constraints encoded in metric
    int w = (sqrt (8* metric + 1) - 1) / 2;
    int t = (w*w + w) / 2;
    int num_cons = metric - t;
    int dom_size = w - num_cons;
    if ( dom_size < other_metric )
    {
        return 1;
    }
    else if ( dom_size == other_metric )
    {
        // Break ties with number of constraints
        if ( num_cons > var->size_constraints_original () )
        {
            return 1;
        }
        else if ( num_cons == var->size_constraints_original () )
        {
            return 0;
        }
        
        return 1;
    }
    else
    {
        return -1;
    }
}//compare

int
MostConstrained::compare ( Variable * var_a , Variable * var_b )
{
    int k1 = (var_a->domain_iterator)->domain_size();
    int k2 = var_a->size_constraints_original ();
    int cantor_pair = ((k1 + k2) * (k1 + k2 + 1) + k2)/2;
    return compare( cantor_pair, var_b );
}//compare

double
MostConstrained::metric_value ( Variable * var )
{
    int k1 = (var->domain_iterator)->domain_size();
    int k2 = var->size_constraints_original ();
    int cantor_pair = ((k1 + k2) * (k1 + k2 + 1) + k2)/2;
    return cantor_pair;
}//metric_value

void
MostConstrained::print () const
{
  std::cout << "most_constrained" << std::endl;
}//print



