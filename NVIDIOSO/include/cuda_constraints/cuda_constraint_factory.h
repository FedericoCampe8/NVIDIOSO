//
//  cuda_constraint_factory.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  Factory method to create instances of constraints on Device (see cuda_constraint.h)
//

#ifndef NVIDIOSO_cuda_constraint_factory_h
#define NVIDIOSO_cuda_constraint_factory_h

#include "base_constraint.h"
#include "cuda_constraint_inc.h"
#include "cuda_global_constraint_inc.h"
#include "cuda_propagation_utilities.h"

#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace CudaConstraintFactory {
#if CUDAON

    /**
     * Factory function for instantiating (base) constraints on device.
     * @param constraint_description array of constraints to instantiate identified by
     *        type
     *        id
     *        scope size
     *        number of auxiliary arguments
     *        identifiers of the variables in the constraint's scope
     *        auxiliary arguments (list) needed by the constraint
     * @param size number of constraints stored in constraint_description
     * @param domain_index indeces of the "beginning" of each variable's domain
     * @param domain_states array containing all domains
     * @param constraint_aux_info pointer to the array of auxiliary information 
     *        needed in order to propagate constraints (e.g., tables, arrays, etc.).   
     */
    __global__ void
    cuda_constrain_factory (  int* constraint_description, size_t size,
                              int* domain_index, uint* domain_states,
                              int* additional_constraint_params_idx,
                              int* additional_constraint_params  )
    {
        // Allocate memory for pointers to constraint instancs
        G_DEV_CONSTRAINTS_ARRAY = (CudaConstraint**) malloc ( size * sizeof ( CudaConstraint* ) );
		
        // Instantiate constriants on device
        int index = 0;
        int c_id;
        int n_vars;
        int n_aux;
        int * vars = nullptr;
        int * args = nullptr;
        int * additional_params = nullptr;
        CudaConstraint * ptr;
        
        /*
         * @Todo Replace the following loop with parallel allocation
         *       to speedup preproccesing/allocation of constraints
         */
        for ( int c = 0; c < size; c++ )
        {
            c_id   			  = constraint_description  [ index + 1 ];
            n_vars 			  = constraint_description  [ index + 2 ];
            n_aux  			  = constraint_description  [ index + 3 ];
            vars   			  = (int*) &constraint_description [ index + 4 ];
            args   			  = (int*) &constraint_description [ index + 4 + n_vars ];
            
            // Set additional parameters needed by the constrain to be propagated
            if ( additional_constraint_params_idx[ c ] >= 0 )
            {
            	additional_params = (int*) &additional_constraint_params [ additional_constraint_params_idx[ c ] ];
            }
            
            switch ( (BaseConstraintType) constraint_description[ index ] )
            {                    
                case BaseConstraintType::ARRAY_BOOL_AND:
                    ptr = new CudaArrayBoolAnd ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::ARRAY_BOOL_ELEMENT:
                    ptr = new CudaArrayBoolElement ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::ARRAY_BOOL_OR:
                    ptr = new CudaArrayBoolOr ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::ARRAY_INT_ELEMENT:
                    ptr = new CudaArrayIntElement ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::ARRAY_SET_ELEMENT:
                    ptr = new CudaArraySetElement ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::ARRAY_VAR_BOOL_ELEMENT:
                    ptr = new CudaArrayVarBoolElement ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::ARRAY_VAR_INT_ELEMENT:
                    ptr = new CudaArrayVarIntElement ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::ARRAY_VAR_SET_ELEMENT:
                    ptr = new CudaArrayVarSetElement ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL2INT:
                    ptr = new CudaBool2Int ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_AND:
                    ptr = new CudaBoolAnd ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_CLAUSE:
                    ptr = new CudaBoolClause ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_EQ:
                    ptr = new CudaBoolEq ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_EQ_REIF:
                    ptr = new CudaBoolEqReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_LE:
                    ptr = new CudaBoolLe ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_LE_REIF:
                    ptr = new CudaBoolLeReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_LT:
                    ptr = new CudaBoolLt ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_LT_REIF:
                    ptr = new CudaBoolLtReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_NOT:
                    ptr = new CudaBoolNot ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_OR:
                    ptr = new CudaBoolOr ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::BOOL_XOR:
                    ptr = new CudaBoolXor ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_ABS:
                    ptr = new CudaIntAbs ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_DIV:
                    ptr = new CudaIntDiv ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_EQ:
                    ptr = new CudaIntEq ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_EQ_REIF:
                    ptr = new CudaIntEqReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LE:
                    ptr = new CudaIntLe ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LE_REIF:
                    ptr = new CudaIntLeReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LIN_EQ:
                    ptr = new CudaIntLinEq ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LIN_EQ_REIF:
                    ptr = new CudaIntLinEqReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LIN_LE:
                    ptr = new CudaIntLinLe ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LIN_LE_REIF:
                    ptr = new CudaIntLinLeReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LIN_NE:
                    ptr = new CudaIntLinNe ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LIN_NE_REIF:
                    ptr = new CudaIntLinNeReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LT:
                    ptr = new CudaIntLt ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_LT_REIF:
                    ptr = new CudaIntLtReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_MAX_C:
                    ptr = new CudaIntMaxC ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_MIN_C:
                    ptr = new CudaIntMinC ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_MOD:
                    ptr = new CudaIntMod ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_NE:
                    ptr = new CudaIntNe ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_NE_REIF:
                    ptr = new CudaIntNeReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_PLUS:
                    ptr = new CudaIntPlus ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::INT_TIMES:
                    ptr = new CudaIntTimes ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_CARD:
                    ptr = new CudaSetCard ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_DIFF:
                    ptr = new CudaSetDiff ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_EQ:
                    ptr = new CudaSetEq ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_EQ_REIF:
                    ptr = new CudaSetEqReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_IN:
                    ptr = new CudaSetIn ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_IN_REIF:
                    ptr = new CudaSetInReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_INTERSECT:
                    ptr = new CudaSetIntersect ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_LE:
                    ptr = new CudaSetLe ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_LT:
                    ptr = new CudaSetLt ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_NE:
                    ptr = new CudaSetNe ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_NE_REIF:
                    ptr = new CudaSetNeReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_SUBSET:
                    ptr = new CudaSetSubset ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_SUBSET_REIF:
                    ptr = new CudaSetSubsetReif ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_SYMDIFF:
                    ptr = new CudaSetSymDiff ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                case BaseConstraintType::SET_UNION:
                    ptr = new CudaSetUnion ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                default:
                    ptr = nullptr;
                    break;
            }
            
            // Sanity check
            if ( ptr == nullptr ) continue;
            
            // Set additional parameters
            if ( additional_params != nullptr )
            {
            	ptr->set_additional_parameters ( static_cast<void *> ( additional_params ) );
            }
            
            // Set the pointer into the global array of constraints
            G_DEV_CONSTRAINTS_ARRAY[ c ] = ptr;

            // Update index for next constraint
            index += 4 + n_vars + n_aux;
        }//c
    }//cuda_constrain_factory
    
    /**
     * Factory function for instantiating global constraints on device.
     * @param constraint_description array of constraints to instantiate identified by
     *        type
     *        id
     *        scope size
     *        number of auxiliary arguments
     *        identifiers of the variables in the constraint's scope
     *        auxiliary arguments (list) needed by the constraint
     * @param size number of constraints stored in constraint_description
     * @param domain_index indeces of the "beginning" of each variable's domain
     * @param domain_states array containing all domains
     * @param constraint_aux_info pointer to the array of auxiliary information 
     *        needed in order to propagate constraints (e.g., tables, arrays, etc.).   
     */
    __global__ void
    cuda_global_constrain_factory (  int* constraint_description, size_t size,
                              		 int* domain_index, uint* domain_states,
                              		 int* additional_constraint_params_idx,
                              		 int* additional_constraint_params  )
    {
        // Allocate memory for pointers to constraint instancs
        G_DEV_GLB_CONSTRAINTS_ARRAY = (CudaConstraint**) malloc ( size * sizeof ( CudaGlobalConstraint* ) );
		
        // Instantiate constriants on device
        int index = 0;
        int c_id;
        int n_vars;
        int n_aux;
        int * vars = nullptr;
        int * args = nullptr;
        int * additional_params = nullptr;
        CudaGlobalConstraint * ptr;
        
        /*
         * @Todo Replace the following loop with parallel allocation
         *       to speedup preproccesing/allocation of constraints
         */
        for ( int c = 0; c < size; c++ )
        {
            c_id   			  = constraint_description  [ index + 1 ];
            n_vars 			  = constraint_description  [ index + 2 ];
            n_aux  			  = constraint_description  [ index + 3 ];
            vars   			  = (int*) &constraint_description [ index + 4 ];
            args   			  = (int*) &constraint_description [ index + 4 + n_vars ];
            
            // Set additional parameters needed by the constrain to be propagated
            if ( additional_constraint_params_idx[ c ] >= 0 )
            {
            	additional_params = (int*) &additional_constraint_params [ additional_constraint_params_idx[ c ] ];
            }
            
            switch ( (GlobalConstraintType) constraint_description[ index ] )
            {                    
                case GlobalConstraintType::ALLDIFFERENT:
                    ptr = new CudaAlldifferent ( c_id, n_vars, n_aux, vars, args, domain_index, domain_states );
                    break;
                default:
                    ptr = nullptr;
                    break;
            }
            
            // Sanity check
            if ( ptr == nullptr ) continue;
            
            // Set additional parameters
            if ( additional_params != nullptr )
            {
            	ptr->set_additional_parameters ( static_cast<void *> ( additional_params ) );
            }
            
            // Set the pointer into the global array of constraints
            G_DEV_GLB_CONSTRAINTS_ARRAY[ c ] = ptr;

            // Update index for next constraint
            index += 4 + n_vars + n_aux;
        }//c
    }//cuda_constrain_factory
    
#endif
  
}//CudaConstraintFactory

#endif
