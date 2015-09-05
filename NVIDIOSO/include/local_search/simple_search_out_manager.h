//
//  simple_search_out_manager.h
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

#ifndef __NVIDIOSO__simple_search_out_manager__
#define __NVIDIOSO__simple_search_out_manager__

#include "search_out_manager.h"

class SimpleSearchOutManager;
typedef std::unique_ptr<SimpleSearchOutManager> SimpleSearchOutManagerUPtr;
typedef std::shared_ptr<SimpleSearchOutManager> SimpleSearchOutManagerSPtr;

class SimpleSearchOutManager : public SearchOutManager {
private:
	
	//! Lookup table mapping string identifiers of search_out_eval(s) with their (unique) id(s)
	std::unordered_map < std::string, std::size_t > _string_eval_lookup;
	
	//! As add_out_evaluator but sets string identifies as well
	void add_out_evaluator ( SimpleSearchOutEvaluator* out_eval, std::string str );
	 
protected:
	
	//! Debug string info
	std::string _dbg;
	
	//! Out value for the search process
	bool _search_out;
	
	//! Initialization function
	virtual void initialize_manager ();
	
	/**
	 * Return true if the evaluator with id "eval_id" is active, false otherwise.
	 * @param eval_id id of the evaluator to query.
	 * @return true if the evaluator with id "eval_id" is active, false otherwise.
	 */
	bool is_active_evaluator ( std::size_t eval_id ) const;
	
public:
	
	SimpleSearchOutManager ();
	
	~SimpleSearchOutManager ();
	
	/**
	 * Notify this search_out_manager about an out event
	 * happend on one of the search_out_evaluators.
	 * @param eval_id id of the search_out_evaluator which has notified this manager
	 */ 
	 void notify_out ( std::size_t eval_id ) override;
	
	/**
	 * Force out on this manager.
	 * @note this method overpass all the evaluators forcing the search to be terminated.
	 */ 
	void force_out () override; 
	
	/**
	 * Reset value for search_out_evaluators (i.e., resets its state).
	 * @param eval_id id of the evaluator to reset.
	 */ 
	void reset_out_evaluator ( std::size_t eval_id ) override;
	
	/**
	 * Reset all values for all search_out_evaluators (i.e., resets their states).
	 * @param eval_id id of the evaluator to reset.
	 * @note reset state of all evaluators even if they are not active.
	 */ 
	 void reset_out_evaluator () override;
	 
	/**
	 * Activate a given (id) out_evaluator
	 * @param eval_id id of the out_evaluator to activate.
	 */
	void activate_out_evaluator ( std::size_t eval_id ) override;
	
	/**
	 * Deactivate a given (id) out_evaluator
	 * @param eval_id id of the out_evaluator to deactivate.
	 */
	void deactivate_out_evaluator ( std::size_t eval_id ) override;
	
	/**
	 * Return true if the search has to be terminated false otherwise.
	 * This method queries every active evaluator to check if they 
	 * have reached their internal limit.
	 * @return true if the search has to be terminated, false otherwise.
	 * @note It resets the search_out internal flag to false.
	 *       By reseting the internal flag, the client can reset other
	 *       evaluators (e.g., iterative improvements steps) to continue
	 *       exploring the search space.
	 */
	bool search_out () override;
	    
	// Set out values
	void set_num_restarts_out ( std::size_t num_sol );
	void set_num_iterative_improvings_out ( std::size_t num_sol );
	void set_num_solutions_out ( std::size_t num_sol );
	void set_time_out ( double timeout );
	void set_num_nodes_out ( std::size_t out_n );
	void set_num_wrong_decisions_out ( std::size_t out_w );
	
	// Update out values
	void upd_restarts ( std::size_t num_sol );
	void upd_iterative_improvings_steps ( std::size_t num_sol );
	void upd_solutions ( std::size_t num_sol );
	void upd_time ( double timeout );
	void upd_nodes ( std::size_t out_n );
	void upd_wrong_decisions ( std::size_t out_w );

	//! Print information about the search_out_manager
	void print() const override;
	
};

#endif /* defined(__NVIDIOSO__simple_search_out_manager__) */
