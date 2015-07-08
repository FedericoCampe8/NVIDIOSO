//
//  constraint_inc.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  Inclusion file to group all inclusions for cuda constraints.
//

#ifndef NVIDIOSO_constraint_inc_h
#define NVIDIOSO_constraint_inc_h

// Constraint base class
#include "constraint.h"

// Constraints iNVIDIOSO v. 1.0
#include "array_bool_and.h"
#include "array_bool_element.h"
#include "array_bool_or.h"
#include "array_int_element.h"
#include "array_set_element.h"
#include "array_var_bool_element.h"
#include "array_var_int_element.h"
#include "array_var_set_element.h"
#include "bool_2_int.h"
#include "bool_and.h"
#include "bool_clause.h"
#include "bool_eq.h"
#include "bool_eq_reif.h"
#include "bool_le.h"
#include "bool_le_reif.h"
#include "bool_lt.h"
#include "bool_lt_reif.h"
#include "bool_not.h"
#include "bool_or.h"
#include "bool_xor.h"
#include "int_abs.h"
#include "int_div.h"
#include "int_eq.h"
#include "int_eq_reif.h"
#include "int_le.h"
#include "int_le_reif.h"
#include "int_lin_eq.h"
#include "int_lin_eq_reif.h"
#include "int_lin_le.h"
#include "int_lin_le_reif.h"
#include "int_lin_ne.h"
#include "int_lin_ne_reif.h"
#include "int_lt.h"
#include "int_lt_reif.h"
#include "int_max_c.h"
#include "int_min_c.h"
#include "int_mod.h"
#include "int_ne.h"
#include "int_ne_reif.h"
#include "int_plus.h"
#include "int_times.h"
#include "set_card.h"
#include "set_diff.h"
#include "set_eq.h"
#include "set_eq_reif.h"
#include "set_in.h"
#include "set_in_reif.h"
#include "set_intersect.h"
#include "set_le.h"
#include "set_lt.h"
#include "set_ne.h"
#include "set_ne_reif.h"
#include "set_subset.h"
#include "set_subset_reif.h"
#include "set_sym_diff.h"
#include "set_union.h"

#endif
