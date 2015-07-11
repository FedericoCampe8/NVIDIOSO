//
//  indomain_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/08/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "indomain_metric.h"

InDomain::InDomain () {
  _dbg = "InDomain - ";
  _metric_type = ValueChoiceMetricType::INDOMAIN;
}//InDomain

int
InDomain::metric_value ( Variable * var )
{
    //! @Todo
    return (var->domain_iterator)->min_val ();
}//metric_value

void
InDomain::print () const
{
    std::cout << "indomain" << std::endl;
}//print
