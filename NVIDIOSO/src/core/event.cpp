//
//  event.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "event.h"

Event::Event () {
  _domain_event = EventType::OTHER_EVT;
}//Event

Event::Event ( EventType domain_event ) {
  _domain_event = domain_event;
}//Event

EventType
Event::get_domain_event() const {
  return _domain_event;
}//get_domain_event