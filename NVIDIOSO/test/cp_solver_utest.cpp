//
//  cp_solver_utest.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/10/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.d.
//

#include "cp_solver_utest.h"

using std::cout;
using std::endl;

CPSolverUTest::CPSolverUTest() :
	UnitTest("CPSolver") {}

CPSolverUTest::~CPSolverUTest() {}

bool
CPSolverUTest::test()
{
	CPModel * model_a = new CPModel ();
	CPModel * model_b = new CPModel ();
	CPModel * model_c = new CPModel ();
	CPModel * model_d = new CPModel ();
	CPSolver * base_cp_solver = new CPSolver ();
	CPSolver * null_cp_solver = new CPSolver ( CPModelUPtr ( nullptr ) );
	CPSolver * model_cp_solver = new CPSolver ( CPModelUPtr ( model_a ) );
	
	// Sanity check 
	if ( model_a == nullptr || model_b == nullptr ||  model_c == nullptr  ||  model_c == nullptr ||
	     base_cp_solver == nullptr || null_cp_solver == nullptr || model_cp_solver == nullptr )
	{
		set_failure_message ( "Instances not created - NULL pointer" );
		delete base_cp_solver;
		delete null_cp_solver;
		delete model_cp_solver;
		return false;
	}
	
	try
	{
		std::size_t zero   = 0;
		std::size_t one    = 1;
		std::size_t two    = 2;
		std::size_t three  = 3;
		
		// Test number of models
		TEST_EQUAL(base_cp_solver->num_models(), zero, "num_models [test 1]");
		TEST_EQUAL(null_cp_solver->num_models(), zero, "num_models [test 2]");
		TEST_EQUAL(model_cp_solver->num_models(), one, "num_models [test 3]");
		
		// Test number of run models 
		TEST_EQUAL(base_cp_solver->num_solved_models(), zero, "num_solved_models [test 1]");
		TEST_EQUAL(null_cp_solver->num_solved_models(), zero, "num_solved_models [test 2]");
		TEST_EQUAL(model_cp_solver->num_solved_models(), zero, "num_solved_models [test 3]");
		
		// Test sat of models
		TEST_EQUAL(base_cp_solver->sat_models(), zero, "sat_models [test 1]");
		TEST_EQUAL(null_cp_solver->sat_models(), zero, "sat_models [test 2]");
		TEST_EQUAL(model_cp_solver->sat_models(), zero, "sat_models [test 3]");
		
		// Test number of models
		TEST_EQUAL(base_cp_solver->unsat_models(), zero, "unsat_models [test 1]");
		TEST_EQUAL(null_cp_solver->unsat_models(), zero, "unsat_models [test 2]");
		TEST_EQUAL(model_cp_solver->unsat_models(), zero, "unsat_models [test 3]");
		
		// ==== Add a model to all CPSolvers ====
		base_cp_solver->add_model  (  CPModelUPtr ( model_b ) );
		null_cp_solver->add_model  (  CPModelUPtr ( model_c ) );
		model_cp_solver->add_model (  CPModelUPtr ( model_d ) );
		
		// Test number of models
		TEST_EQUAL(base_cp_solver->num_models(),  one, "num_models [test 2]");
		TEST_EQUAL(null_cp_solver->num_models(),  one, "num_models [test 3]");
		TEST_EQUAL(model_cp_solver->num_models(), two, "num_models [test 4]");
		
		// Test number of run models 
		TEST_EQUAL(base_cp_solver->num_solved_models(), zero, "num_solved_models [test 2]");
		TEST_EQUAL(null_cp_solver->num_solved_models(), zero, "num_solved_models [test 3]");
		TEST_EQUAL(model_cp_solver->num_solved_models(), zero, "num_solved_models [test 4]");
		
		// Test sat of models
		TEST_EQUAL(base_cp_solver->sat_models(), zero, "sat_models [test 2]");
		TEST_EQUAL(null_cp_solver->sat_models(), zero, "sat_models [test 3]");
		TEST_EQUAL(model_cp_solver->sat_models(), zero, "sat_models [test 4]");
		
		// Test number of models
		TEST_EQUAL(base_cp_solver->unsat_models(), zero, "unsat_models [test 2]");
		TEST_EQUAL(null_cp_solver->unsat_models(), zero, "unsat_models [test 3]");
		TEST_EQUAL(model_cp_solver->unsat_models(), zero, "unsat_models [test 4]");
		
		// ==== Remove a model to all CPSolvers ====
		base_cp_solver->remove_model  (  model_b->get_id () );
		null_cp_solver->remove_model  (  model_c->get_id () );
		model_cp_solver->remove_model (  model_d->get_id () );
		
		// Test number of models
		TEST_EQUAL(base_cp_solver->num_models(),  zero, "num_models [test 3]");
		TEST_EQUAL(null_cp_solver->num_models(),  zero, "num_models [test 4]");
		TEST_EQUAL(model_cp_solver->num_models(), one, "num_models [test 5]");
		
		// Test number of run models 
		TEST_EQUAL(base_cp_solver->num_solved_models(), zero, "num_solved_models [test 3]");
		TEST_EQUAL(null_cp_solver->num_solved_models(), zero, "num_solved_models [test 4]");
		TEST_EQUAL(model_cp_solver->num_solved_models(), zero, "num_solved_models [test 5]");
		
		// Test sat of models
		TEST_EQUAL(base_cp_solver->sat_models(), zero, "sat_models [test 3]");
		TEST_EQUAL(null_cp_solver->sat_models(), zero, "sat_models [test 4]");
		TEST_EQUAL(model_cp_solver->sat_models(), zero, "sat_models [test 5]");
		
		// Test number of models
		TEST_EQUAL(base_cp_solver->unsat_models(), zero, "unsat_models [test 3]");
		TEST_EQUAL(null_cp_solver->unsat_models(), zero, "unsat_models [test 4]");
		TEST_EQUAL(model_cp_solver->unsat_models(), zero, "unsat_models [test 5]");
		
		// ==== Get Model ====
		TEST_EQUAL( const_cast<CPModel *> ( model_cp_solver->get_model( model_a->get_id() ) ), model_a, "get_model [test 1]" );
		TEST_EQUAL( model_cp_solver->get_model( model_a->get_id() )->get_id (), model_a->get_id(),      "get_model [test 2]" );
		
		// ==== Run Model ====
		model_cp_solver->run ();
		TEST_EQUAL(model_cp_solver->num_solved_models(), one, "num_solved_models [test 6]");
		TEST_EQUAL(model_cp_solver->sat_models(), zero, "sat_models [test 6]");
		TEST_EQUAL(model_cp_solver->unsat_models(), zero, "unsat_models [test 6]");
		
		model_cp_solver->run ( model_a->get_id() );
		TEST_EQUAL(model_cp_solver->num_solved_models(), two, "num_solved_models [test 7]");
		TEST_EQUAL(model_cp_solver->sat_models(), zero, "sat_models [test 7]");
		TEST_EQUAL(model_cp_solver->unsat_models(), zero, "unsat_models [test 7]");
		
		// ==== Copy Constructor ====
		CPSolver copy_cp_solver ( *model_cp_solver );
		TEST_EQUAL(copy_cp_solver.num_solved_models(), two, "num_solved_models [test 8]");
		TEST_EQUAL(copy_cp_solver.sat_models(), zero, "sat_models [test 8]");
		TEST_EQUAL(copy_cp_solver.unsat_models(), zero, "unsat_models [test 8]");
		TEST_EQUAL( (copy_cp_solver.get_model( model_a->get_id() ))->get_id (), model_a->get_id(), "get_model [test 3]" );
			
		copy_cp_solver.run ();
		TEST_EQUAL(copy_cp_solver.num_solved_models(), three, "num_solved_models after run on copy");
		TEST_EQUAL(model_cp_solver->num_solved_models(), two, "num_solved_models on original after run on copy");
		
		// ==== Assignment operator ====
		CPSolver op_cp_solver;
		op_cp_solver = *model_cp_solver;
		TEST_EQUAL(op_cp_solver.num_solved_models(), two, "num_solved_models [test 9]");
		TEST_EQUAL(op_cp_solver.sat_models(), zero, "sat_models [test 9]");
		TEST_EQUAL(op_cp_solver.unsat_models(), zero, "unsat_models [test 9]");
		TEST_EQUAL( (op_cp_solver.get_model( model_a->get_id()))->get_id (), model_a->get_id(), "get_model [test 4]" );
		
		op_cp_solver.run ();
		TEST_EQUAL(op_cp_solver.num_solved_models(), three, "num_solved_models after run on =copy");
		TEST_EQUAL(model_cp_solver->num_solved_models(), two, "num_solved_models on original after run on =copy");
	}
	catch ( ... )
	{
		set_failure_message ( "Thrown exception not know" );
		delete base_cp_solver;
		delete null_cp_solver;
		delete model_cp_solver;
		return false;
	}
	
	// Clean
	delete base_cp_solver;
	delete null_cp_solver;
	delete model_cp_solver;
	
	return true;
}//run_test
