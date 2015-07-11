//
//  cp_variable.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class defines a FD variable and related operations on it.
//  This is a subject observed by Constraints which are observers.
//  Everytime a variables is modified, such modification is notified to
//  the correspondent list of observers/constraints.
//

#ifndef NVIDIOSO_variable_h
#define NVIDIOSO_variable_h

#include "globals.h"
#include "domain.h"
#include "domain_iterator.h"

class ConstraintStore;
typedef std::shared_ptr<ConstraintStore> ConstraintStorePtr;

class Constraint;
typedef std::shared_ptr<Constraint> ConstraintPtr;

class Variable;
typedef std::shared_ptr<Variable> VariablePtr;

enum class VariableType {
  FD_VARIABLE,
  SUP_VARIABLE,
  OBJ_VARIABLE,
  OTHER
};

class Variable {
protected:
  std::string _dbg;
  
  /**
   * The constraint store on which
   * this variable operates (i.e., constraint store to notify).
   */
  ConstraintStorePtr _constraint_store;
  
  /*
   * Variable ids and info:
   * _id    : global id w.r.t. all the other model objects
   * _str_id: string id of the variable (e.g. x_1)
   */
  int _id;
  std::string  _str_id;
  VariableType _var_type;
  
  //! Total number of observers
  size_t _number_of_constraints;
  
  /**
   * List of constraints attached to this variable.
   * These constraints are organized by the type of event
   * they are triggered by.
   */
  std::map< EventType, std::vector<ConstraintPtr> > _attached_constraints;
  
  /**
   * List of ids of detached constraints from this variable.
   * These ids (i.e., constraints' ids) will be used
   * to restore the variable's state during search.
   * @note |_observer| + |_detach_observers| = _number_of_observers.
   */
  std::list<size_t> _detach_constraints;
  
  /**
   * It checks whether a given id belongs
   * to the list of detached constraints.
   * @param c_id the id of the constraint to check
   *        if it is detached or not.
   * @return true if c_id is attached, i.e., it does not
   *         belong to the list of detached constraints.
   */
  virtual bool is_attached ( size_t c_id );
  
  /**
   * It notifies all the constraints attached to this variables
   * that a change has been done on this very variable.
   */
  virtual void notify_constraint ();
  
  /**
   * It notifies the current store attached to this variable
   * that a change has been done on this very variable.
   * It actually checks which constraint should be reevaluated
   * according to the event happened on the domain.
   */
  virtual void notify_store ();
  
  /**
   * Base constructor.
   * @note a global unique id is assigned to this variable.
   */
  Variable ();
  
  /**
   * Base constructor.
   * @param v_id the id to assign to this variable.
   */
  Variable ( int v_id );
  
public:
  
  virtual ~Variable ();
  
  /**
   * Iterator to use to get domain's elements from
   * the current variable's domain.
   * Domains should be accessed only through this iterator.
   */
  DomainIterator * domain_iterator;
  
  //! Get integer id of this variable.
  int get_id () const;
  
  /**
   * Set the (string) id of the variable.
   * @param str the string to set as variable's identifier.
   */
  void set_str_id ( std::string str );
  
  /**
   * Get the string id of this variable.
   * @return a string representing the id of this variable.
   */
  std::string get_str_id () const;
  
  //! Set the type of variable (i.e., FD_VARIABLE, SUP_VARIABLE, etc.)
  void set_type ( VariableType vt );
  
  //! Get the type of variable (i.e., FD_VARIABLE, SUP_VARIABLE, etc.)
  VariableType get_type () const;
  
  //! Get the event happened on this domain.
  virtual EventType get_event () const = 0;
  
  //! Reset default event on this domain.
  virtual void reset_event () = 0;
  
  /**
   * Set domain according to the specific
   * variable implementation.
   * @note: different types of variable
   * @param dt domain type of type DomainType to set
   *        to the current variable
   */
  virtual void set_domain_type ( DomainType dt ) = 0;
  
  /**
   * It returns the size of the current domain.
   * @return the size of the current variable's domain.
   */
  virtual size_t get_size () const = 0;
  
  /**
   * It checks if the domain contains only one value.
   * @return true if the the variable's domain is a singleton,
   *         false otherwise.
   */
  virtual bool is_singleton () const = 0;
  
  /**
   * It checks if the domain is empty.
   * @return true if variable domain is empty.
   *         false otherwise.
   */
  virtual bool is_empty () const = 0;
  
  //! Print domain
  virtual void print_domain () const = 0;
  
  /**
   * Set a constraint store as current constraint store
   * for this variable.
   * The store will be notified when this variable will change
   * its internal state.
   * @param store the constraint store to attach to this variable.
   */
  virtual void attach_store ( ConstraintStorePtr store );
  
  /**
   * It registers constraint with this variable, so always when this
   * variable is changed the constraint is reevaluated/notified.
   * @param c the (pointer to) the constraint which is added to 
   *        this variable.
   */
  virtual void attach_constraint ( ConstraintPtr c );
  
  /**
   * It detaches constraint from this variable, so change in variable
   * will not cause constraint reevaluation.
   * @param c the (pointer to) the constraint which is detached from
   *        this variable.
   * @note If c appears only to be attached to this variable, 
   *       this method actually destroyes the constraint c. 
   *       The client must be care of storing c somewhere else
   *       in order to restore the state (e.g. for backtrack actions).
   */
  virtual void detach_constraint ( ConstraintPtr c );
  
  /**
   * It detaches constraint from this variable, so change in variable
   * will not cause constraint reevaluation.
   * @param c the id of the constraint which is detached from
   *        this variable.
   * @note If c appears only to be attached to this variable,
   *       this method actually destroyes the constraint c.
   *       The client must be care of storing c somewhere else
   *       in order to restore the state (e.g. for backtrack actions).
   */
  virtual void detach_constraint ( size_t c_id );
  
  /**
   * It notifies the current observers attached to this variable
   * that a change has been done on this very variable.
   */
  virtual void notify_observers ();
  
  /**
   * It returns the current number of constraints attached to
   * this variable and that are not yet satisfied.
   * @return number of constraints attached to the variable
   *         not yet satisfied.
   * @note use this method to implement some heuristics (e.g., 
   *       min conflict heuristic.
   */
  virtual size_t size_constraints ();
  
  /**
   * It returns the current number of constraints attached to
   * this variable (either satisfied or not satisfied yet).
   * @return number of constraints attached to the variable.
   */
  virtual size_t size_constraints_original () const;
  
  //! Print info about the variable
  virtual void print () const;
};


#endif
