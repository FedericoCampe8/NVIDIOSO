//
//  system_description.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#ifndef NVIDIOSO_system_description_h
#define NVIDIOSO_system_description_h


/**
 * @mainpage
 **iNVIDIOSO**
======================================
NVIDIa-based cOnstraint SOlver v. 1.0 
======================================

iNVIDIOSO1.0 is a constraint solver for constraint satisfaction and optimization problems.
It takes advantage of the GPU architecture -- if available -- to speedup exploration 
of the search space and propagation of constraints.
The solver allows the user to choose between complete (e.g., DFS), 
and incomplete search strategies, e.g., (e.g., local search, MCMC sampling, etc.).
iNVIDIOSO1.0 is written in C++ and uses the CUDA programming model for GPU computation.   



Installation
-------------

To install iNVIDIOSO1.0,  first set the environment variables 
in the file
	> NVIDIOSO/config/iNVIDIOSO.envs
	
In this file you can set several environment variables such as the path where 
to install iNVIDIOSO1.0, the compiler version, and the compute capability of 
the graphic card if present.

After the environment variables have been set, you just need to go into the iNVIDIOSO 
main folder run the installation script as follows:
```
$ cd iNVIDIOSO/NVIDIOSO
$ ./install.sh
```
During the installation process you will be asked whether you prefer to install 
the CPU version or the GPU version of the solver:
just type "C" for the CPU version or "G" for the GPU version and press enter.

If everything goes well, you should see the following message at the
end of the installation:
> Installation completed

If the installation cannot be completed successfully, some information about the 
possible issue can be found in the log file[^mail]
[^mail]: Please, feel free to send the log file and/or ask for further 
explanation or suggestions to fede.campe@gmail.com.


> install_hhhhmmdd.log
 
 Please, note that this is an ongoing project.
 In particular, there might be bugs or issues we are not yet aware of. 
 We try to do our best to keep the solver updated and to fix bugs as soon as we find them. 
If you are experiencing an issue, you want to know more about the solver, 
or you have some comments on it, please feel free to send an email to
> fede.campe@gmail.com

Here there are some new features we will implement in the next future:
> - Allow the user to define different local search strategies
     in a declarative way as input for the solver.
     The solver will automatically run the local search strategy in parallel on the GPU;
> - Use a multi-GPU environment to scale horizontally when 
     searching for sub-optimal solutions for optimization problems;
>- Implement the Simplex algorithm (sequential and parallel version) for solving LP problems;
>- Handle Floating point variables and continuous constraints.

Please, refer to the *manual* of the solver to get more information about iNVIDIOSO1.0. 
The folder
> iNVIDIOSO/Doxygen

contains the descriptions of all the classes. 
This folder has been generated using *Doxygen*.

To navigate through the classes, open the index file with your preferred browser:
> iNVIDIOSO/Doxygen/html/index.html

The same content is printed in a pdf file named *iNVIDIOSO_refman.pdf*:
> iNVIDIOSO/Doxygen/iNVIDIOSO_refman.pdf
 
 ----------

Solver's Framework
-------------
  
  We briefly describe how iNVIDIOSO internally represents **Variables**, **Domains**, 
  and **Constraints** in what follows. 

#### <i class="icon-pencil"></i> Variables
iNVIDIOSO1.0 can handle three types of variables
> - Boolean variables: True/False
> - Integer variables: ..., -2, -1, 0, 1, 2, ...
> - Sets of integers: {}, {2, 3, 4}, {1..10}

Internally, iNVIDIOSO1.0 distinguishes between three different variable "objects", namely:
: FD variables: Finite Domain variables;
: SUP Variables: SUPport variables introduced to calculate the objective function. 
These variables have unbounded int domains;
: OBJ Variables: OBJective variables. These variables store the objective value 
as calculated by the objective function through standard propagation. 
These variables have unbounded int domains.

#### <i class="icon-pencil"></i> Domains
iNVIDIOSO1.0 represents Integer and Set domains a sequence 
of 5 Integer values followed by an array of bits.
Boolean domains are represented only by two Integer values.
For Integer and Set variables, domains have the following structure:
>  | EVT | REP | LB | UB | DSZ || ... BIT ... | 

Where:
>- EVT: represents the EVenT happened on the domain and it can be one of the following events:
	>>-  FAILED: for empty domains
	>>- SINGLETON: for domains with just one element
	>>- MIN: when the *lower bound* of the domain has changed due to propagation
	>>- MAX: when the *upper bound* of the domain has changed due to propagation
	>>- BOUND: when the *lower* and/or the *upper* have been changed due to propagation 
	(this is a more general case w.r.t. the MIN, and MAX events)
	>>- CHANGED: when the domain is changed due to propagation ***and***, 
	no bounds are modified and the domain is neither SINGLETON nor FAILED.

>- REP: is the REPresentation used for the domain's elements. 
This field can assume positive and negative values with the following meanings:
>>- -1, -2, -3, ...: the BIT field contains a set of 1, 2, 3, ... bitmaps respectively. 
Each of this bitmaps is stored in the BIT field as a sequence of two Integer values 
representing the lower and upper bounds respectively and a sequence of bits 
representing the elements contained between the two bounds.
>>- 0: the whole BIT field represents a bitmap of (contiguous) values
>>- 1, 2, 3, ...   : the BIT field contains 0, 1, 2, ... 
lists of pairs <*lower*, *upper*> bounds. If the value is 1, then the pair of 
bounds is represented by the two fields LB and UB.

>- LB: Lower Bound of the domain;
>- UB: Upper Bound of the domain;

>- DSZ: Domain SiZe, i.e., number of elements in the domain.
	If DSZ is less than a predefined value, the representation automatically switches 
	to a list of bits representing the domain's elements (i.e., REP = 0).
	
> **Note:**
>- Domains are represented slightly different on GPU. In particular, to save space, 
iNVIDIOSO1.0 represents domains on the GPU ***only*** by using a bitmap representation 
(i.e., REP = 0) ***or*** by a single pair of bounds, i.e., using two 
Integers values when domains cannot be represented by all the bits in the BIT field.


#### <i class="icon-pencil"></i> Constraints
We implemented (almost) all the FlatZinc constraints on integers.
The list of FlatZinc constraints can be found at http://www.minizinc.org/.

 
Thank you for reading this page.
  If you've have any question, suggestion or observation, please don't hesitate to write at 
  > fede.campe@gmail.com. 
  
  --------------------
  
  Why “iNVIDIOSO”?
  “INVIDIA” is an italian word which means “envy”. “INVIDIOSO” 
  is used to refer to someone who is envious for some reason. 
  Moreover, the solver uses the NVIDIA CUDA programming model, 
  that allows one to program code running on NVIDIA graphic cards.
Therefore,
>  i + NVIDI + OSO = iNVIDIA-based cOnstraint SOlver
 */

#endif
