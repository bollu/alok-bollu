Files
-----

- `word2vec`: vanilla word2vec



- `word2vec-grad`: partial attempt to train
word2vec with gradient descent. 



- `word2complex`: train a complex valued network using
 perceptron update

- `word2ga`: train an embedding into a geometric algebra space with a novel non-commutative, bilinear product structure that allows us to measure the amount of one geometric algebra object into another. The problem is that it is quite expensive to compute (requires a traversal over all subspaces). I hope that this can be made faster by either writing good GPU code with a dumb algorithm, or by using a smart algorithm to compute this data. In particular, some variant of XOR convolution seems promising. 

- `word2ff`: train with weighs as integers 
over a finite field. This can be significantly
changed: what we should do is to accumulate gradients
Till they grow to an integer, at which point we should update
the value and truncate the gradient



- `word2man`: try to learn the manifold 
structure more explicitly, rather than
assuming euclidian vectors. currently,
just uses a fixed hyperbolic manifold with
half of the dimensions negative and half
positive. Somewhat sensitive to initialisation.


- `glove`: direct clone of the glove repository


- `glove2man`: glove with a similar manifold training regime


# problems with training as a manifold

The first problem is that word2vec uses perceptron style 
updates and not gradient descent, so this needs to be changed 

More seriously, I hypothesize (and have some experimental evidence
from the python implementation) that the training
suffers from "GAN-style collapse" where either
the random vectors quickly force the metric to become degenerate,
or the metric grows to strongly favour a small number of
dimensions. I do not know how to stabilize this training

Perhaps something like fixing the weights of either the vectors
or the manifold for a batch might work.

Otherwise, read the literature from GAN training

There is the other fact that the repo is fast because the
gradient descent is hand written: the gradient computation
is done by hand, and no automatic differentiation library is used 
I tried to use ATEN because I don't know how to
implement reimannian SGD, but it imposes an order of magnitude performance
cost which is very steep. Unfortunately, I can find no
accessible explanation of reimannian stochastic gradient
descent `:/` I am hoping to be able to read enough diffgeo
so I can implement a version of the algorithm that I
understand.

Similarly, trying to exploit GPU parallelism with
Arrayfire was also useless, since the overhead of
synchronisation kills any performance gains 

In general, the code is written in a style that is extremely
race heavy into its main data structures, which makes adapting
it to any library difficult. 

At some point, I might write CUDA code by hand to try
and parallelise the computation.


Thoughts
--------

consider the cost function of w2v:
 ` cost = label - <focus|ctx>`


Let us consider a perfect analogy, with all values
taken from text8 on 200 dimensions
```
1.   france - paris ~ india - delhi
    <france|france> - <france|paris> ~ <france|india> - <france|delhi>
    1 - 0.63 ~? 0.43 - 0.24 
    0.37 ~? 0.19


    <paris|france> - <paris|paris> ~ <paris|india> - <paris|delhi>
    0.63 - 1 ~? 0.23 - 0.32
    -0.37 ~? -0.09


    <india|france> - <india|paris> ~ <india|india> - <india|delhi>
    0.43 - 0.23 1 ~? 1 - 0.61
    0.2 ~? 0.4

    <delhi|france> - <delhi|paris> ~ <delhi|india> - <delhi|delhi>
    0.24 - 0.32 ~? 0.61 - 1
    -0.08 ~? -0.39
```

Clearly, the regime of using word vectors for cosine similarity AND
that of word analogy does not really pan out, due to the differences
in delta. So, what we should aim to do is to find alterante representations
which play well with both of these concepts.



We might want to learn an affine space, where POINT is a set,
DIR a vector space that can act on POINT. 
word analogy: `king - man + woman`
as: `ACT(DIR(king, man), woman)`

the question is, what the fuck is a legit training scheme for this?

### Glove

Let us look at glove derivation: `c` for context representation, `w`
for regular representation

```
F(w_i, w_j, c_k) = P_ik / P_jk
F((w_i - w_j)^T c_k) = P_ik / P_jk -- (1)
```
Note that eqn (1) is absurd: It is measuring `c_k` along the path from
`w_j` to `w_i`. What does that even mean????


Next, we ask that `F` is a homomorphism from `(R, +)` to `(R, x)`,
or `F = exp`, giving:
```
F(w_i, w_j, c_k) = P_ik / P_jk
F((w_i - w_j)^T c_k) 
  = F((w_i^T c_k) - (w_j^T c_k)) 
  = F(w_i^T c_k) / F(w_j^T c_k) -- (2)
```

`(1) = (2)` gives:

```
F(w_i^T c_k) / F(w_j^T c_k) = P_ik / P_jk
```

so one is naturally tempted to set:

```
F(w_a^T c_b) = P_ab
exp(w_a^T c_b) = P_ab = X_ab / X_a (WHAT? isn't this P(b|a) ?)
w_a^T c_b = log(P_ab)
```

Hence, we arrive at the conclusion that the dot product should be
the log prob.

## "Abstract Glove"


Let `C_i ∈ N^VOCABSIZE`.
`C_i[j]` be the number of times word `j` occurs in the context of word `i`.
the sum of entries of `C_i` will denote the number of times any word appears in
the context of `C_i`.

First notice that this set of arrays `C = { C_i }` has no a priori reasonable
mathematical structure. They are not closed under vector addition, do 
not have an identity vector, etc. 

What we wish to do is to extract an algebraic embedding out of these objects.

Let the group `(G, .)` be the embedding of the analogies among elements of `C`.
Let the embedding be denoted as `g_c ∈` G for `c ∈ C`, and let the
action of `G` on `C` be denote as `=>: G x C -> C`

Let `g_a, g_b ∈ G`. We will denote multiplication as `g_a g_b`, and inverses
as `g_a^`. We will write `g_a g_b^` as `g_a - g_b`, but please note that
this is only notational convenience.

That is, for words `c_i:c_j :: c_k:?`, we can answer the question of what
`?` is by compute `(g_j - g_i) => c_k`. 

If the group `(G, .)` is fixed, then the objects we need to learn are:
- The mappings `c -> g_c`
- The group action `=>: G x C -> C`

Let `diff_g_c: G x C -> R` rank how close `C` is to the spirit of `G`? One
choice is something like:

```
diff_g_a_c: g_a
```

