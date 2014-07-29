//
//  event.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents events that happened during the solving process.
//  These events can be used to infer different kinds of knowledge about
//  the solving process, e.g., which constraints need to be propagated
//  after a given type of event.
//

#ifndef __NVIDIOSO__event__
#define __NVIDIOSO__event__

#include "globals.h"
#include "domain.h"

class Event {
protected:
  EventType _domain_event;
  
public:
  Event ();
  Event ( EventType domain_event );
  
  virtual EventType get_domain_event() const;
  
};

#endif /* defined(__NVIDIOSO__event__) */
