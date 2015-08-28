//
//  fzn_search_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/10/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Factory class for search method according to FlatZinc the specifications.
//

#ifndef __NVIDIOSO__search_generator__
#define __NVIDIOSO__search_generator__

#include "globals.h"
#include "depth_first_search.h" 
#include "simple_local_search.h"
#include "simple_heuristic.h"
#include "indomain_search_initializer.h"
#include "iterated_conditional_modes_heuristic.h"
#include "metric_inc.h"
#include "simple_solution_manager.h"
#include "simple_backtrack_manager.h"
#include "neighborhood_backtrack_manager.h"
#include "simple_search_out_manager.h"

class SearchFactory {
public:
  
	/**
   	 * Get the right instance of FlatZinc search method
   	 * according to its type described by the input string.
   	 * @param variables a vector of pointers to all the variables to label.
   	 * @param search_tkn reference to a search token in order to
   	 *        define the right instance of search engine.
   	 */
	static  SearchEnginePtr get_search_shr_ptr ( std::vector< Variable * > variables, TokenSol * search_tkn, Variable * obj_var = nullptr ) 
	{
		// Sanity check
  	  	assert ( search_tkn != nullptr );
    
      	// Create a new search engine and set it up
      	SearchEnginePtr engine;
      
      	// Set heuristic, backtrack manager, and solution manager
      	HeuristicPtr heuristic;
      	VariableChoiceMetric * var_choice_metric = nullptr;
      	ValueChoiceMetric    * val_choice_metric = nullptr;
      	
      	// Backtrack Manager
      	BacktrackManagerPtr backtrack_manager;
    
		// Var goal
      	std::string var_goal = search_tkn->get_var_goal ();
      	if ( var_goal != "" )
      	{
      		LogMsg << "Search Engine - Goal variable: " + var_goal << std::endl;
      	}
    
      	std::string solve_goal = search_tkn->get_solve_goal ();
      	if ( solve_goal != "satisfy" )
      	{
        	LogMsg << "Solve goal choice not yet implement: " + solve_goal << std::endl;
      	}
      	LogMsg << "Search Engine - Solve goal set: satisfy" << std::endl;
      
      	// Search choice
      	std::string search_choice = search_tkn->get_search_choice ();
      	if ( ( search_choice != "int_search" ) && ( search_choice != "") )
      	{
          LogMsg << "Search choice not yet implement: " + search_choice << std::endl;
          std::string str_err = "Error in search choice settings\n";
          throw NvdException ( str_err.c_str(), __FILE__, __LINE__ );
      	}
      	LogMsg << "Search Engine - Search choice set: " + search_choice << std::endl;
      
      	std::string label_choice = search_tkn->get_label_choice ();
      	if ( label_choice != "" )
      	{
        	LogMsg << "Search Engine - Search labeling performed on variable: " << label_choice << std::endl;
      	}
      
      	std::vector< std::string > vars_to_label = search_tkn->get_var_to_label();
      	if ( vars_to_label.size() != 0 )
      	{
      		LogMsg << "Search Engine - Search labeling performed on variables: " << std::endl;
      		for ( auto& s: vars_to_label )
      			LogMsg << s << " ";
      		LogMsg << std::endl;
      	}
      	
		// Search strategy
		std::string strategy_choice = search_tkn->get_strategy_choice();
		
      	// Set variable selection heuristic
      	std::string variable_choice = search_tkn->get_variable_choice();
      	if ( strategy_choice == "complete" && variable_choice == "" )
      	{
      		variable_choice = "input_order";
      	}
      	if ( variable_choice != "" )
      	{
        	if ( variable_choice  == "input_order" )
          	{
            	var_choice_metric = new InputOrder ();
          	}
          	else if ( variable_choice == "first_fail" ) {
            	var_choice_metric = new FirstFail ();
          	}
          	else if ( variable_choice == "anti_first_fail" ) {
            	var_choice_metric = new AntiFirstFail ();
          	}
          	else if ( variable_choice == "smallest" ) {
            	var_choice_metric = new Smallest ();
          	}
          	else if ( variable_choice == "largest" ) {
              	var_choice_metric = new Largest ();
          	}
          	else if ( variable_choice == "occurence" ) {
              	var_choice_metric = new Occurence ();
          	}
          	else if ( variable_choice == "most_constrained" ) {
              	var_choice_metric = new MostConstrained ();
          	}
          	else if ( variable_choice == "max_regret" ) {
              	var_choice_metric = new MaxRegret ();
          	}
          	else
          	{
              	LogMsg << "Search variable choice not implemented: " + variable_choice << std::endl;
              	std::string str_err = "Search variable choice not implemented: " + variable_choice + "\n";
              	throw NvdException ( str_err.c_str(), __FILE__, __LINE__ );
          	}
          	LogMsg << "Search Engine - Search variable choice set: " + variable_choice << std::endl;
      	}

      	// Set assignment heuristic for variables
      	std::string assignment_choice = search_tkn->get_assignment_choice();
      	if ( strategy_choice == "complete" && assignment_choice == "" )
      	{
      		variable_choice = "indomain_min";
      	}
      	if ( assignment_choice != "" ) 
      	{
        	if ( assignment_choice == "indomain_min" )
          	{
            	val_choice_metric = new InDomainMin ();
          	}
          	else if ( assignment_choice == "indomain_max" )
          	{
              	val_choice_metric = new InDomainMax ();
          	}
          	else if ( assignment_choice == "indomain_median" )
          	{
              	val_choice_metric = new InDomainMedian ();
          	}
          	else if ( assignment_choice == "indomain" )
          	{
              	val_choice_metric = new InDomain ();
          	}
          	else if ( assignment_choice == "indomain_random" )
          	{
              	val_choice_metric = new InDomainRandom ();
          	}
          	/*
          	else if ( assignment_choice == "indomain_split" )
          	{
              	val_choice_metric = new InDomainSplit ();
          	}
          	else if ( assignment_choice == "indomain_reverse_split" )
          	{
              	val_choice_metric = new InDomainReverseSplit ();
          	}
          	else if ( assignment_choice == "indomain_interval" )
          	{
              	val_choice_metric = new InDomainInterval ();
          	}
          	*/
          	else
          	{
              	LogMsg << "Assignment value choice not implemented: " + assignment_choice << std::endl;
              	std::string str_err = "Assignment value choice not implemented: " + assignment_choice + "\n";
              	throw NvdException ( str_err.c_str(), __FILE__, __LINE__ );              
          	}
          	LogMsg << "Search Engine - Assignment value choice set: " + assignment_choice << std::endl;
      	}
    
      	if ( strategy_choice == "complete" )
      	{
      		engine = std::make_shared<DepthFirstSearch> ();
      		heuristic = std::make_shared<SimpleHeuristic>( variables, var_choice_metric, val_choice_metric );
      		backtrack_manager = std::make_shared<SimpleBacktrackManager>();
      	}
      	else if ( strategy_choice == "incomplete" || strategy_choice == "local_search" || strategy_choice == "heuristic" )
      	{
      		int var_goal_index = -1;
      		if ( var_goal != "" )
      		{
      		
      		} 
      		engine = std::make_shared<SimpleLocalSearch> ();
      		heuristic = std::make_shared<IteratedConditionalModesHeuristic>( variables, obj_var );
      		
      		(std::dynamic_pointer_cast<SimpleLocalSearch> (engine))->
      		set_search_initializer ( std::unique_ptr<InDomainSearchInitializer> ( new InDomainSearchInitializer ( variables ) ) );
      		
      		(std::dynamic_pointer_cast<SimpleLocalSearch> (engine))->
      		set_search_out_manager ( std::make_shared<SimpleSearchOutManager> () );
      		backtrack_manager = std::make_shared<NeighborhoodBacktrackManager>();
      	}
      	else
      	{
      		LogMsg << "Search strategy choice not implemented: " + strategy_choice << std::endl;
        	std::string str_err = "Assignment value choice not yet implemented: " + strategy_choice + "\n";
            throw NvdException ( str_err.c_str(), __FILE__, __LINE__ );
      	}
      	
      	// Set Heuristic on the search engine
      	engine->set_heuristic( heuristic );
    
      	for ( auto var : variables )
      	{// Attach each variable to the backtrack manager
      		
        	//(std::static_pointer_cast<SimpleBacktrackManager>( backtrack_manager ))
        	backtrack_manager->
            attach_backtracable( static_cast<IntVariable*>(var) );
            
          	static_cast<IntVariable*>(var)->set_backtrack_manager( backtrack_manager );
      	}
    
      	// Set backtrack manager on search engine.
      	engine->set_backtrack_manager( backtrack_manager );
    
      	// Solution Manager
      	SolutionManager * solution_manager = new SimpleSolutionManager ( variables );
    
      	// Set solution manager on search engine.
      	engine->set_solution_manager( solution_manager );

      	return engine;
  	}
};

#endif /* defined(__NVIDIOSO__search_generator__) */
