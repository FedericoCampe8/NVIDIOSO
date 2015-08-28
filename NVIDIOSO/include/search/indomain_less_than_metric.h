//
//  indomain_less_than_metric.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements the indomain_min metric for value choice.
//

#ifndef __NVIDIOSO__indomain_less_than_metric__
#define __NVIDIOSO__indomain_less_than_metric__
 
#include "value_choice_metric.h"

class InDomainLessThan : public ValueChoiceMetric {
private:

	//! Lookup table <var id, value> for metric_value ( Variable* ) method.
	std::unordered_map< std::size_t, int > _current_val_lookup;
	
public:
  InDomainLessThan ();
   
  /**
   * Gets value to assign to var using greater_than choice.
   * @param var the (pointer to) variable for which a value if needed.
   * @param comparator integer value to compare to return a strictly less than value.
   * @return the value to assign to var.
   * @note if no value is available it returns comparator.
   * @note this method will also SET comparator in _current_val_lookup.
   */
  int metric_value ( Variable * var, int comparator ) override;
  
  /**
   * Gets value to assign to var using greater_than choice.
   * @param var the (pointer to) variable for which a value if needed.
   * @return the value to assign to var.
   * @note this method returns a value strictly less than previous value at each invokation.
   * @note if no value is available it returns the max domain's element and resets _current_val_lookup. 
   */
  int metric_value ( Variable * var ) override;
  
  void print () const;
};

#endif /* defined(__NVIDIOSO__indomain_less_than_metric__) */
