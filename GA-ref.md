## CLT for Generalised Manifolds(A VERY short Summary)

https://arxiv.org/pdf/1801.06581.pdf <p>
This paper provides a generalised CLT on manifolds and aims to make the concept of **Smeariness** (A phenomenon which manifests only in non-euclidean geometry) precise. 
The CLT theorem is proved by extending a previous work which proved the theorem for Frechet functions. The baisc idea was to use certain properties like intrinsic means(Generalized Frechet means in there case) which apparently paints a much better picture of a probability model we know very little about, makes a bunch of reasonable assumptions (atleast this is what it looked like) and then extends it to the general case. 

Honestly, smeariness is just extra information which I found pretty interesting, might be useful later. In simple terms, without using measure theory concepts, it goes like that - A sequence of random vectors $X_n$ is k-th order smeary if $n^{\frac{1}{2(k+1)}}X_n$ has a non-trivial limiting distributions as $n \to \infty$. For instance, the fluctuation of random directions on the circle of sample means around the population mean may feature smeariness of any given positive integer order.


## An attempt to use directional statistics with GA

Now that we know central limit theorem holds even for general manifolds, it just boils down to finding the equivalent of the very familiar euclidean normal distribution on the manifold we choose. This would help us frame a formula for $P(u|v)$ where $u$ is context word representation and $v$ is focus vector representation. Before that, a little bit of primer and motivation for using GA.

### Why Directional Statistics?
Directional statistics constitute an important tool for processing manifold-valued data. It deals with observations, which are not familiar counts, real numbers or (unrestricted) vectors, but they are directions in 2- or 3-dimensional space or rotations of such space. A direction can be regarded as a multi-dimensional unit vector. Such vectors form directional data. Examples of directional data are manifold-valued data, spherical and shape data. In parametric models of directional data, appropriate probability density functions must be defined on directional data. 

Thus when dealing with manifolds and stuff, the probabilistic model will be a directional model. As far as I have read and observed, the directional model in its parametric form can be described loosely as follows -
$$G(K)*exp(Metric(X,K))$$ where $K$ = some parameter , $G(K)$ = some normalizing constant

Here is an example of oen such parametric models - 

1. Matrix-Fisher model - In case of hypersphere $(S^{n-1})$, we can have a probability function $P$ such that $P(x|a) = \alpha(a)exp(a^Tx)$ where $a \in R^n$ is a parameter vector and $\alpha(a) = \frac{||a||^{d/2-1}}{\Gamma(d/2)I_{d/2}(||a||)}$. Here $\Gamma(.)$ is the Gamma function and $I_d(.)$ is the modified bessel function of first kind and order $d$. But how did such a big function come here arbitratrily?
The general way to go about it is to integrate some well defined density function over the entire manifold which will be very hard to calculate in certain manifolds(even in the case of sphere, it requires a lot of obnoxious integration tricks). So instead of having to do such leg work, a smarter way is to approximate the integral in question by using some sort of series. In the case of hyperspher, it turns out to be the Bessel Function.
This model can also be extended to Steifel Manifold. The equation becomes - 
$$P(X|A) = \frac{1}{\alpha(A)}exp\{tr(X^TA)\}$$ where $A = R^{d\times p}$ and $\alpha(.)$ is the normalizing constant.

Keeping all this in mind, we look at our GA problem. Our proposed GA method involves viewing each word as a subspace $R^k$ where $k\leq$ EMBED DIMENSION

Thus, the closest model to this GA space has to be the one of those used for Grassmanian manifold. I choose the matrix bingham model, seems the most convenient. An alternative could be Angular Central Gaussian model. 
The matrix bingham model goes something like this.

$$P(X|B)=\frac{1}{C(B)}exp\{tr(X^TBX)\}$$ ($X^TX = I_d$, $B$ =  symmetric $d\times d$ parameter matrix) 

Now using series approximation, we get $C(.) = {}_1F_1(d/2,p/2,B)$. Here ${}_1F_1$ is a hypergeometric function with a matrix argument.

### Challenges 


1. I am very confused about how to relate $tr(X^TBX)$ with our inner product ($A_r\rfloor B_s$). Is it even possible? Because the GA equivalent of trace is the scalar product thing which turned out to be a glorified inner product.  

2. As each word uses different subspaces, doesn't this mean the value of B will be different depending on the word in question. This results in actual calculation of the normalizing constant during the objective function formulation? Need to find an effecient way to code hypergeometric functions.


