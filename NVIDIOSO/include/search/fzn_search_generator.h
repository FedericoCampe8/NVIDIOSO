//
//  fzn_search_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 10/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Factory class for search method according to FlatZinc the specifications.
//

#ifndef __NVIDIOSO__fzn_search_generator__
#define __NVIDIOSO__fzn_search_generator__

#include "globals.h"
#include "depth_first_search.h"
#include "simple_heuristic.h"
#include "input_order_metric.h"
#include "first_fail_metric.h"
#include "indomain_min_metric.h"
#include "indomain_max_metric.h"
#include "simple_solution_manager.h"
#include "simple_backtrack_manager.h"

class FZNSearchFactory {
public:
  
  /**
   * Get the right instance of FlatZinc search method
   * according to its type described by the input string.
   * @param variables a vector of pointers to all the variables in the model.
   * @param search_tkn reference to a search token in order to
   *        define the right instance of search engine.
   */
  static  SearchEnginePtr get_fzn_search_shr_ptr ( std::vector< Variable * > variables,
                                                   TokenSol * search_tkn ) {
    // Create a new search engine to set up
    SearchEnginePtr engine = std::make_shared<DepthFirstSearch>();
    
    // Set heuristic, backtrack manager, and solution manager.
    HeuristicPtr heuristic;
    VariableChoiceMetric * var_choice_metric = nullptr;
    ValueChoiceMetric    * val_choice_metric = nullptr;
    
    // Var goal
    std::string var_goal = search_tkn->get_var_goal ();
    if ( var_goal.compare ( "" ) != 0 )
      std::cout << "Goal variable choice not yet implement: " + var_goal << std::endl;
    
    std::string solve_goal = search_tkn->get_solve_goal ();
    if ( solve_goal.compare ( "satisfy" ) != 0 )
      std::cout << "Solve goal choice not yet implement: " + solve_goal << std::endl;
    
    std::string search_choice = search_tkn->get_search_choice ();
    if ( (search_choice.compare("int_search") != 0) &&
         (search_choice.compare("") != 0) )
      std::cout << "Search choice not yet implement: " + var_goal << std::endl;
    
    std::string label_choice = search_tkn->get_label_choice ();
    if ( var_goal.compare ( "" ) != 0 ) {
      std::cout << "Search on all variables.\n";
      std::cout << "Search on subset of variables not yet implemented: ";
      std::cout << label_choice << std::endl;
    }

    std::string variable_choice = search_tkn->get_variable_choice();
    if ( variable_choice.compare ( "" ) != 0 ) {
      if ( variable_choice.compare ( "input_order" ) == 0 ) {
        var_choice_metric = new InputOrder ();
      }
      else if ( variable_choice.compare ( "first_fail" ) == 0 ) {
        var_choice_metric = new FirstFail ();
      }
      else {
        std::cout << "Search variable choice not yet implemented: " <<
        variable_choice << std::endl;
      }
    }
    
    std::string assignment_choice = search_tkn->get_assignment_choice();
    if ( assignment_choice.compare ( "" ) != 0 ) {
      if ( assignment_choice.compare ( "indomain_min" ) == 0 ) {
        val_choice_metric = new InDomainMin ();
      }
      else if ( assignment_choice.compare ( "indomain_max" ) == 0 ) {
        val_choice_metric = new InDomainMax ();
      }
      else {
        std::cout << "Assignment value choice not yet implemented: " <<
        assignment_choice << std::endl;
      }
    }
    
    std::string strategy_choice = search_tkn->get_strategy_choice();
    
    // Default variable choice: InputOrder
    if ( var_choice_metric == nullptr ) var_choice_metric = new InputOrder ();
    
    // Default value choice: InDomainMin
    if ( val_choice_metric == nullptr ) val_choice_metric = new InDomainMin ();
    
    // Set heuristics
    heuristic = std::make_shared<SimpleHeuristic>( variables,
                                                   var_choice_metric,
                                                   val_choice_metric );
    // Set Heuristic on search engine.
    engine->set_heuristic( heuristic );
    
    // Backtrack Manager
    BacktrackManagerPtr backtrack_manager = std::make_shared<SimpleBacktrackManager>();
    for ( auto var : variables ) {
      (std::static_pointer_cast<SimpleBacktrackManager>( backtrack_manager ))->
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

#endif /* defined(__NVIDIOSO__fzn_search_generator__) */
