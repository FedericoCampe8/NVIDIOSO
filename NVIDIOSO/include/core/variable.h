//
//  cp_variable.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class defines a FD variable and related operations on it.
//

#ifndef NVIDIOSO_variable_h
#define NVIDIOSO_variable_h

#include "globals.h"
#include "domain.h"

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
  
  /**
   * Pointer to the domain of the variable.
   * @note: each variable is associated with a Finite Domain.
   */
  DomainPtr _domain_ptr;
  
public:
  Variable ();
  Variable ( int );
  virtual ~Variable ();
  
  // Get/Set methods
  int get_id () const;
  
  /**
   * Set the (string) id of the variable.
   * @param str the string to set as variable's identifier
   */
  void        set_str_id ( std::string str );
  std::string get_str_id () const;
  
  void         set_type ( VariableType vt );
  VariableType get_type () const;
  
  /**
   * Set domain according to the specific
   * variable implementation.
   * @note: different types of variable
   * @param dt domain type of type DomainType to set
   *        to the current variable
   */
  virtual void set_domain_type ( DomainType dt );
  
  //! Print info about the variable
  virtual void print () const = 0;
};


#endif
