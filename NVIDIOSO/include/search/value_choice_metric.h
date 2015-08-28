//
//  value_choice_metric.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface for the value choice metrics
//  that can be used for defining a heuristic.
//  This interface is used to compare values given a variable, i.e.,
//  it is an interface for different indomain enumeration methods.
//


#ifndef __NVIDIOSO__value_choice_metric__
#define __NVIDIOSO__value_choice_metric__

#include "globals.h"
#include "variable.h"

enum class ValueChoiceMetricType {
  INDOMAIN_MIN,
  INDOMAIN_MAX,
  INDOMAIN_MIDDLE,
  INDOMAIN_MEDIAN,
  INDOMAIN,
  INDOMAIN_RANDOM,
  OTHER
};

class ValueChoiceMetric {
protected:
  //! Debug string
  std::string _dbg;
  
  //! Value choice metric type
  ValueChoiceMetricType _metric_type;
  
  ValueChoiceMetric ();

public:
  virtual ~ValueChoiceMetric();
  
  /**
   * Get the type of metric for this value choice metric.
   * @return the metric type of this value choice metric.
   */
  virtual ValueChoiceMetricType metric_type () const;
  
  /**
   * Auxiliary function needed when the metric value to get depends
   * on an externel value (e.g., "greater_than" metric).
   * @param var (pointer to) the variable for which value for assignment is given.
   * @param comparator integer value to compare with the current metric value.
   * @return the value to assign to the given variable.
   * @note by default this method invokes metric_value ( var ).
   */
  virtual int metric_value ( Variable * var, int comparator );
  
  /**
   * Returns the value within a variable's domain which should be
   * used to label the current variable.
   * @param var (pointer to) the variable for which value for assignment is given.
   * @return the value to assign to the given variable.
   */
  virtual int metric_value ( Variable * var ) = 0;
  
  //! Print info about this value choice metric
  virtual void print () const = 0;
};

#endif /* defined(__NVIDIOSO__value_choice_metric__) */
