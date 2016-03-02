**iNVIDIOSO**
===================
iNVIDIa-based cOnstraint SOlver v. 1.0 
===============================

iNVIDIOSO1.0 is the back-end solver for optimization problems.
It takes advantage of the GPU architecture -- if available -- to speedup exploration of the search space and propagation of constraints.
The solver allows the user to choose between complete (e.g., DFS), and incomplete search strategies, e.g., (e.g., local search, MCMC sampling, etc.).
iNVIDIOSO1.0 is written in C++ and uses the CUDA programming model (developer.nvidia.com) for GPU computation.  
Based on parallel and distributed computation iNVIDIOSO1.0 is able to solve large scale real-world problems.
The back end solver has a front end language and a GUI to easily create models and run them interactively.


DISCLAIMER
-------------

Update 1/26/2016
-----------------
iNVIDIOSO1.0 is almost ready for its first beta-release! We are currently finishing to implement the last features for the upcoming release
and testing both the language and the back-end.
We are almost there.


-----------------

The source code of iNVIDIOSO1.0 has been removed from github.
We are implementing new features, search strategies and parallel algorithms to improve iNVIDIOSO performance 
and enhance user experience.
We are also fixing some major bugs.
We will distribute a beta version of iNVIDIOSO1.0 soon, before its first release.

New features we will implement in the next future:
> - Allow the user to define different local search strategies
     in a declarative way as input for the solver.
     The solver will automatically run the local search strategy in parallel on the GPU;
> - Use a multi-GPU environment to scale horizontally when 
     searching for sub-optimal solutions for optimization problems;
>- Implement the Simplex algorithm (sequential and parallel version) for solving LP problems;
>- Handle variables on reals and continuous constraints;
>- GUI.
 
Thank you for reading this page.
  If you've have any question, suggestion or observation, please don't hesitate to write at 
  > fede.campe@gmail.com. 
  
  --------------------
  
  Why “iNVIDIOSO”?
  “INVIDIA” is an italian word which means “envy”. “INVIDIOSO” is used to refer to someone who is envious for some reason. Moreover, the solver uses the NVIDIA CUDA programming model, that allows one to program code running on NVIDIA graphic cards.
Therefore,
>  i + NVIDI + OSO = iNVIDIA-based cOnstraint SOlver

