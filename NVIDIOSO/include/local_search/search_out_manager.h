//
//  search_out_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class declares the interface for a search out manager.
//  A search out manager maps each search position and memory state 
//  to a probability distribution over truth values, 
//  which indicates the probability with which the search is to be terminated 
//  upon reaching a specific search position and memory state.
//  In general, the user can set different parameters to terminate the (local) search
//  process, such as: timeout, number of wrong decisions, number of restarts, 
//  number of iterative improvements, epsilon on the objective value, etc..
//

#ifndef __NVIDIOSO__search_out_manager__
#define __NVIDIOSO__search_out_manager__

#include "globals.h"
#include "simple_search_out_evaluator.h"

class SearchOutManager;
typedef std::unique_ptr<SearchOutManager> SearchOutManagerUPtr;
typedef std::shared_ptr<SearchOutManager> SearchOutManagerSPtr;

class SearchOutManager {
protected: 

	/**
	 * Hash table of out evaluators, where
	 * - key   = out_evaluator unique id
	 * - value = pair <out_evaluator active (Boolean value), (pointer to) out_evaluator (with id key)>
	 */
	std::unordered_map < std::size_t, std::pair<bool, SimpleSearchOutEvaluator* > > _out_evaluators;	
	 
public:
	virtual ~SearchOutManager () {}
	
	/**
	 * Set an out value for the given search_out_evaluator.
	 * @param eval_id (unique) id of the evaluator to set
	 * @param val value to set for the eval_id evaluator
	 */
	template<typename T> void 
	set_out_value ( std::size_t eval_id, T val ) 
	{
		auto it = _out_evaluators.find ( eval_id );
		if ( it != _out_evaluators.end () )
		{
			(_out_evaluators [ eval_id ].second)->set_limit_out ( val );
		}
	}//set_value_out
	 
	/**
	 * Update an out value for the given search_out_evaluator.
	 * @param eval_id (unique) id of the evaluator to set
	 * @param val value to updtae for the eval_id evaluator
	 */
	template<typename T> void 
	upd_metric_value ( std::size_t eval_id, T val ) 
	{
		auto it = _out_evaluators.find ( eval_id );
		if ( it != _out_evaluators.end () )
		{
			(_out_evaluators [ eval_id ].second)->upd_metric_value ( val );
		}
	}//set_value_out 
	
	/**
	 * Get the out value for the given search_out_evaluator.
	 * @param eval_id (unique) id of the evaluator to query
	 * @return val value get for the eval_id evaluator
	 */
	template<typename T> T
	get_out_value ( std::size_t eval_id ) const 
	{
		auto it = _out_evaluators.find ( eval_id );
		if ( it == _out_evaluators.end () )
		{
			std::string err_msg { "SearchOutManager::get_out_value SearchOutEvaluator not found" }; 
			throw NvdException ( err_msg.c_str() );
		}
		
		return ((_out_evaluators.at ( eval_id )).second)->get_limit_out ();
	}//get_out_value
	
	/**
     * Add a new out_evaluator to the current set of out_evaluator.
     * @param out_eval (unique) pointer to an out_evaluator
     * @note When an out_evaluator is added, it is automatically activated.
     */
     void add_out_evaluator ( SimpleSearchOutEvaluator* out_eval )
	 {
	 	_out_evaluators [ out_eval->get_id () ] = std::make_pair( true, out_eval );
	 }//add_out_evaluator
	   
	/**
	 * Notify this search_out_manager about an out event
	 * happend on one of the search_out_evaluators.
	 * @param eval_id id of the search_out_evaluator which has notified this manager.
	 * @note this method is used to implement the Observer pattern.
	 *       It can be implemented as an empty method if no observer is required.
	 */ 
	 virtual void notify_out ( std::size_t eval_id ) = 0;
	 
	/**
	 * Force out on this manager.
	 * @note this method overpass all the evaluators forcing the search to be terminated.
	 */ 
	 virtual void force_out () = 0;
	  
	/**
	 * Reset value for search_out_evaluators (i.e., resets its state).
	 * @param eval_id id of the evaluator to reset .
	 */ 
	 virtual void reset_out_evaluator ( std::size_t eval_id ) = 0;
	 	  
	 /**
	 * Reset all values for all search_out_evaluators (i.e., resets their states).
	 * @param eval_id id of the evaluator to reset .
	 */ 
	 virtual void reset_out_evaluator () = 0;
	
	/**
	 * Activate a given (id) out_evaluator
	 * @param eval_id id of the out_evaluator to activate.
	 */
	virtual void activate_out_evaluator ( std::size_t eval_id ) = 0;
	
	/**
	 * Deactivate a given (id) out_evaluator
	 * @param eval_id id of the out_evaluator to deactivate.
	 */
	virtual void deactivate_out_evaluator ( std::size_t eval_id ) = 0;
	
	/**
	 * Return true if the search has to be terminated 
	 * false otherwise.
	 * @return true if the search has to be terminated, false otherwise.
	 */
	virtual bool search_out () = 0;
	
	//! Print information about the search_out_manager
	virtual void print() const = 0;
};

#endif /* defined(__NVIDIOSO__search_out_manager__) */
