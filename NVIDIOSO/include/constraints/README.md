<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

**iNVIDIOSO**
===================
iNVIDIOSO1.0 Constraints 
===============================
**iNVIDIOSO1.0** implement several *base* and *global* constraints.
The following is a list of the current available base constraints.

> **Note:**
> - Variables are indicated by letters (*a*, *b*, *c*, etc.)
> - Arrays have size of *n*;
> - When two arrays are present, usually *as* is the array of variables and *bs* the array of constant values;
> - Base constraints implement the (sub)set of *FlatZinc* constraints. For further information, please refer to www.minizinc.org.

| Constraint   |  Semantic  | 
----------------- | ------------------
| array_bool_element  | $as[b] = c$ | 
| bool_2_int  | $a = b$ |
| int_eq  | $a = b$ | 
| int_le  | $a \leq b$ | 
| int_lin_eq  | $\sum_{i \in 1..n} as[i].bs[i] = c$ | 
| int_lin_ne  | $\sum_{i \in 1..n} as[i].bs[i] \neq c$ | 
| int_lt  | $a < b$ |
| int_ne  | $a \neq b$ |
| int_plus  | $a + b = c$ |
| int_times  | $a * b = c$ |

Thank you for reading this page, for any further question, please feel free to contact fede.campe@gmail.com. 

