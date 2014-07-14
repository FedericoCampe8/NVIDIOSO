//
//  cp_domain.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "domain.h"

using namespace std;

Domain::Domain () :
_dom_type ( DomainType::OTHER ) {
}//Domain

Domain::~Domain() {}//~Domain

void
Domain::set_type ( DomainType dom_type ) {
  if ( _dom_type == DomainType::OTHER ) {
    _dom_type = dom_type;
  }
}//set_type

DomainType
Domain::get_type () const {
  return _dom_type;
}//get_type




