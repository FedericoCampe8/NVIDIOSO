//
//  indomain_median_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "indomain_median_metric.h"

InDomainMedian::InDomainMedian () {
  _dbg = "InDomainMedian - ";
  _metric_type = ValueChoiceMetricType::INDOMAIN_MEDIAN;
}//InDomainMin

int
InDomainMedian::metric_value ( Variable * var )
{
    //! @Todo
    return (var->domain_iterator)->min_val ();
}//metric_value

void
InDomainMedian::print () const
{
    std::cout << "indomain_median" << std::endl;
}//print
