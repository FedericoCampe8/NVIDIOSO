//
//  local_search.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class define the interface class for local search strategies.
//  It extends the SearchEngine class with other methods and classes needed by
//  Local Search strategies
//

#ifndef __NVIDIOSO__local_search__
#define __NVIDIOSO__local_search__

#include "search_engine.h"
#include "local_search_heuristic.h"
#include "search_initializer.h"
#include "search_memory_manager.h"
#include "search_out_manager.h"

class LocalSearch : public SearchEngine {
protected:
  
  	LocalSearch () {};
  
public:

	virtual ~LocalSearch () {};
  
  	/**
   	 * Set the initializer used to define the initial search positions.
   	 * This defines the initialization of the search process. 
   	 * @param a reference to a local search initializer.
   	 */
  	virtual void set_search_initializer ( SearchInitializerUPtr initializer ) = 0;
  
  	/**
   	 * Set the manager used by local search algorithms to determine when 
   	 * the search is to be terminated upon reaching a specific search position 
   	 * (and/or memory state).
   	 * @param a reference to a search memory manager object.
   	 */
  	virtual void set_search_out_manager ( SearchOutManagerSPtr search_out_manager ) = 0;
};

#endif /* defined(__NVIDIOSO__local_search__) */
