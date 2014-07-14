//
//  search_engine.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface for a search engine.
//  Different search strategies implement this interface.
//

#ifndef NVIDIOSO_search_engine_h
#define NVIDIOSO_search_engine_h

#include "globals.h"

class SearchEngine;
typedef std::shared_ptr<SearchEngine> SearchEnginePtr;

class SearchEngine {
  
public:
  
  SearchEngine ();
  virtual ~SearchEngine ();
  
  
};



#endif
