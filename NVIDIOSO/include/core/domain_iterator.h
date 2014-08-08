//
//  domain_iterator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Iterator's interface that allows the client to
//  explore the domain of a given variable.
//

#ifndef __NVIDIOSO__domain_iterator__
#define __NVIDIOSO__domain_iterator__

#include "globals.h"
#include "int_domain.h"

class DomainIterator {
protected:
  IntDomainPtr _domain;
  
public:
  DomainIterator ( IntDomainPtr domain );
  
  virtual ~DomainIterator();
  
  /**
   * It returns the current minimal value in domain.
   * @return the minimum value belonging to the domain.
   */
  virtual int min_val    () const;
  
  /**
   * It returns the current maximal value in domain.
   * @return the maximum value belonging to the domain.
   */
  virtual int max_val    () const;
  
  /**
   * It returns a random value from domain.
   * @return the a random value belonging to the domain.
   */
  virtual int random_val () const;
  
};

#endif /* defined(__NVIDIOSO__domain_iterator__) */
