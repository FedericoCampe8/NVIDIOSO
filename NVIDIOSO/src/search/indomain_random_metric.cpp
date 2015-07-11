//
//  indomain_random_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "indomain_random_metric.h"

InDomainRandom::InDomainRandom () {
    _dbg = "InDomainRandom - ";
    _metric_type = ValueChoiceMetricType::INDOMAIN_RANDOM;
}//InDomainRandom

int
InDomainRandom::metric_value ( Variable * var )
{
    return (var->domain_iterator)->random_val ();
}//metric_value

void
InDomainRandom::print () const
{
    std::cout << "indomain_random" << std::endl;
}//print
