//
//  global_constraint.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 30/07/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class specializes the Constraint class for 
//  global constraints.
//  The interested reader can refer to the global constraint catalog
//      http://sofdem.github.io/
//  for the complete list of global constraints and their specification.
//
//  @note To implement a constraint at least the following methods
//        should be specialized:
//        - constructor
//        - setup method
//        - consistency
//        - satisfied
//        - print_semantic
//

#ifndef __NVIDIOSO__global_constraint__
#define __NVIDIOSO__global_constraint__

#include "constraint.h"

class GlobalConstraint;
typedef std::shared_ptr<GlobalConstraint> GlobalConstraintPtr;

enum class GlobalConstraintType
{
  ALLDIFFERENT            = 0,
  OTHER                   = 1
};

class GlobalConstraint : public Constraint {
protected:
  
  //! Scope size
  int _scope_size;
  
  //! Constraint type
  GlobalConstraintType _global_constraint_type;
  
  //! Array of (base/global) constraints involved in this global constraint
  std::vector<Constraint *> _sub_constraint;
  
  /*
   * === Information for parallel implementation (following CUDA model) ===
   * Following members are not needed for sequential implementation.
   * However, they can be used as information for a parallel
   * implementation of this global constraint.
   * ======================================================================
   */

  //! Number of parallel blocks (i.e., groups of threads) 
  std::size_t _num_blocks;
  
  //! Number of parallel threads per block
  std::size_t _num_threads;
  
  //! Size in bytes of (local) memory required (e.g., shared memory)
  std::size_t _local_memory;
  
  //! Set global constraint type (default GlobalConstraintType::OTHER)
  void set_global_constraint_type ( GlobalConstraintType gbl_t );
  
  //! Naive consistency algorithm
  virtual void naive_consistency ();
  
  //! Bound consistency algorithm
  virtual void bound_consistency ();
  
  //! Full consistency algorithm
  virtual void full_consistency ();
  
  /**
   * Base constructor.
   * @param name the name of the global constraint.
   */
  GlobalConstraint ( std::string name );
  
public:
  
  virtual ~GlobalConstraint();
  
  //! Get global constraint type
  GlobalConstraintType get_global_constraint_type () const;
  
  //! Number of blocks of threads needed for parallel implementation
  std::size_t get_num_blocks () const;
  
  //! Number of threads per block needed for parallel implementation
  std::size_t get_num_threads () const;
  
  //! Size in bytes of local memory to allocate for parallel implementation
  std::size_t get_size_memory () const;
  
  /**
   * It sets the variables and the arguments for this constraint.
   * @param vars a vector of pointers to the variables in the
   *        constraint's scope.
   * @param args a vector of strings representing the auxiliary
   *        arguments needed by the constraint in order to ensure
   *        consistency.
   */
  virtual void setup ( std::vector<VariablePtr> vars, std::vector<std::string> args );
  
  /**
   * It attaches this constraint (observer) to the list of
   * the variables in its scope.
   * When a variable changes state, this constraint could be
   * automatically notified (depending on the variable).
   */
  void attach_me_to_vars () override;
  
  /**
   * It is a (most probably incomplete) consistency function which
   * removes the values from variable domains. Only values which
   * do not have any support in a solution space are removed.
   */
  void consistency () override;
  
  /**
   * It checks if the constraint is satisfied.
   * @return true if the constraint if for certain satisfied,
   *         false otherwise.
   * @note If this function is incorrectly implementd,
   *       a constraint may not be satisfied in a solution.
   */
  bool satisfied () override;
  
  /**
   * It removes the constraint by removing this constraint
   * from all variables in its scope.
   */
  void remove_constraint ();
  
  //! Prints info.
  void print () const override;
  
  //! Prints the semantic of this constraint.
  void print_semantic () const override;
};

#endif /* defined(__NVIDIOSO__global_constraint__) */
