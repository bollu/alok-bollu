Title
-----
The main aim of this project is to figure out a way to implement the word2vec using a semi-definite program solver.

# Reasoning

The reasoning for attempting such goes below:

# Problem formulation


focus vectors: `f_i`
context vectors: `c_j`

- For words that appear in the context: minimize `|1 - f_i . c_j|` 

- For words that do not appear in the context: minimize `|0 - f_i . c_j|` 

# Solvers that can be used

- [`scipopt`](https://scip.zib.de/) can be downloaded and installed. The have
  generic `.deb` archives that appear to work on Ubuntu ([Link here](https://scip.zib.de/download.php?fname=SCIPOptSuite-6.0.2-Linux.de))

- [`mosek`](https://www.mosek.com/downloads/) provides free licenses for 30
  days. We can use this for now as well.



# Reformulation of the problem 
- Matrix C for now will be set as 0. We really don't care about minimising an objective at the moment.
- So the real question is just describing each A<sub>k</sub>.
    Also, word vector matrix X will have elements of the form `x_i_j = f_i . c_j`  