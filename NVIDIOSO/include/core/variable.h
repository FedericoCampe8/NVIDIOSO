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

class Constraint;
typedef std::shared_ptr<Constraint> ObserverPtr;


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
  
  /*
   * Variable ids and info:
   * _id    : global id w.r.t. all the other model objects
   * _str_id: string id of the variable (e.g. x_1)
   */
  int _id;
  std::string  _str_id;
  VariableType _var_type;
  
  //! Total number of observers
  size_t _number_of_observers;
  
  /**
   * List of observers of this variable.
   * These observers (i.e., constraints) will be notified
   * when a variable is changed.
   */
  std::list<ObserverPtr> _observers;
  
  /**
   * List of ids of detached observers from this variable.
   * These ids (i.e., constraints' ids) will be used
   * to restore the variable's state during search.
   * @note |_observer| + |_detach_observers| = _number_of_observers.
   */
  std::list<size_t> _detach_observers;
  
public:
  Variable ();
  Variable ( int );
  virtual ~Variable ();
  
  //! Get integer id of this variable
  int get_id () const;
  
  /**
   * Set the (string) id of the variable.
   * @param str the string to set as variable's identifier
   */
  void        set_str_id ( std::string str );
  std::string get_str_id () const;
  
  void         set_type ( VariableType vt );
  VariableType get_type () const;
  
  //! Get (const) reference to this domain
  virtual const DomainPtr domain () = 0;
  
  //! Get event on this domain
  virtual EventType get_event () const = 0;
  
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
  
  /**
   * It registers constraint with this variable, so always when this
   * variable is changed the constraint is reevaluated/notified.
   * @param c the (pointer to) the constraint which is added to 
   *        this variable.
   */
  virtual void attach_constraint ( ObserverPtr c );
  
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
  virtual void detach_constraint ( ObserverPtr c );
  
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
   * It notifies all the constraints attached to this variables
   * that a change has been done on this very variable.
   */
  virtual void notify_constraint ();
  
  /**
   * It notifies the current store attached to this variable
   * that a change has been done on this very variable.
   */
  virtual void notify_store ();
  
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
