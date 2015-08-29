//
//  local_search.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class define the interface class for local search strategies.
//  It extends the SearchEngine class with other methods and classes needed by
//  Local Search strategies.
//  @note this simple interface extends the more general interface of SearchEngine.
//        Local Search Heuristics, Backtrack managers, etc. are set  
//        by invoking SearchEngine's standard methods.
//        This interface adds some other default parameters and methods specific
//        to Local Search strategies such as: iterative improving limit, 
//        number of restarts to perform, search initializer object, etc. 
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
  	
  	/**
   	 * Set the maximum number of iterative improving steps to perform.
   	 * An Iterative Improving (II) step is a restart of the (local) search strategy
   	 * where the starting point is the best solution found so far.
   	 * @param ii_limit unsigned value representing the maximum number of 
   	 *        iterative improving steps to perform (default 0 II).
   	 */
  	virtual void set_iterative_improving_limit ( std::size_t ii_limit=0 ) = 0;
  	
  	/**
   	 * Set the maximum number of restarts to perform.
   	 * A restarts calles the (local) search strategy again, starting from 
   	 * the initial solution provided to the strategy (e.g., by the 
   	 * search initializer).
   	 * @param restarts_limit unsigned value representing the maximum number of 
   	 *        restarts to perform (default 0 restarts).
   	 * @note A restart is not performed if some other limit has been already reached,
   	 *       e.g., a restart is not performed if timeout limit has been reached.
   	 */
  	virtual void set_restarts_limit ( std::size_t restarts_limit=0 ) = 0;
};

#endif /* defined(__NVIDIOSO__local_search__) */
