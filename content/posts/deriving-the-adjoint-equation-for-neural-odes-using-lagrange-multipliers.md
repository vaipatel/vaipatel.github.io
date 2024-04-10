+++
title = "Deriving the Adjoint Equation for Neural ODEs using Lagrange Multipliers"
date = "2020-02-04T07:18:43"
tags = ["Machine Learning", "Lagrange Multipliers", "Neural ODEs"]
keywords = ["Machine Learning", "Neural ODEs"]
categories = ["Machine Learning"]
slug = "deriving-the-adjoint-equation-for-neural-odes-using-lagrange-multipliers"
draft = false
showtoc = true
tocopen = false
+++

A Neural ODE ​[^1]​ expresses its output as the solution to a [dynamical system](https://en.wikipedia.org/wiki/Dynamical_system) whose evolution function is a learnable neural network. In other words, a Neural ODE models the transformation from input to output as a learnable ODE.

Since our model is a learnable ODE, we use an ODE solver to evolve the input to an output in the forward pass and calculate a loss. For the backward pass, we would like to simply store the function evaluations of the ODE solver and then backprop through them to calculate the loss gradient. Unfortunately, the forward solve might have taken a huge number of steps depending on the system dynamics, possibly making our computation graph too big to hold in memory for backprop.

Luckily, a very well-known technique called the Adjoint Sensitivity method [^2]​ helps retrieve the gradient of the Neural ODE loss with respect to the network parameters by solving another ODE backwards in time, sidestepping the need to store intermediate function evaluations.

The adjoint method is used by the Neural ODE authors who also provide a very intuitive proof for it in the paper’s Appendix (B.1), alternate to the one presented in [^2]​.

Nevertheless, I couldn’t find a copy of [^2]​, so here I take a stab at deriving the adjoint method using what I think is used in ​[^2]– the more traditional approach of [Lagrange Multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier).

For kicks.⚽

(My attempt relies heavily on my readings of ​[^3], [^​4]​, and [^​5​].)

Problem
-------

First let’s set the stage by formulating our problem and breaking it into subproblems.

### PM: Minimization Problem

Here’s our main minimization problem.
$$\tag{PM} \text{Find } {\underset {\theta}{\operatorname {argmin} }} \enskip L(z(t_1))$$
$$\text{subject to}$$

$$\tag{1} F(\dot{z(t)}, z(t), \theta, t) = \dot{z(t)} – f(z(t), \theta, t) = 0$$
$$\tag{2} z(t_0) = z_{t_0}$$
$$t_0 < t_1$$

$\text{(PM)}$'s dramatis personae a.k.a variables are as follows.

*   $f$ is our neural network with parameters $\theta$
*   $z(t_0)$ is our input, $z(t_1)$ is the output, $z(t)$ is the state reached from $z(t_0)$ at time $t \in \[t_0, t_1\]$
*   $\dot{z(t)} = \frac{\mathrm{d}z(t)}{\mathrm{d}t}$ is the time derivative of our state $z(t)$.
*   $L$ is our loss and its a function of the output $z(t_1)$ .

In the forward pass, we solve the initial value problem given by $(1)$-$(2)$ to go from input to output, like so
$$\tag{3} z(t_1) = z_{t_0} + \int_{t_0}^{t_1} f(z(t), \theta, t)dt$$

### PG: Grad Sub-Problem

$\text{(PM)}$ is a constrained non-linear optimization problem. To tackle it with gradient descent, we need the gradient of the loss $L(z(t_1))$ with respect to the network params $\theta$. But we saw in the intro that simple backprop can be memory inefficient because of the potentially huge computation graph. So we want to
$$\tag{PG} \text{Efficiently calculate } \frac{\mathrm{d} L(z(t_1))}{ \mathrm{d} \theta}$$

But how can we make the grad calculation more memory-efficient? We would need to somehow do a backward pass without storing all the function activations from the forward pass. Perhaps we can leverage the ODE constraint given by $(1)$?

### PL: Lagrangian Sub-Sub-Problem

To incorporate our ODE constraint $\text{(1)}$ into $\text{(PM)}$ we will use the technique of [Lagrange Multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier). In particular, let’s entwine our original our ODE $F(\dot{z(t)}, z(t), \theta, t)=0$ into our original objective $L(z(t_1))$ to make a new objective like so
$$\tag{4} \psi = L(z(t_1)) - \int_{t_0}^{t_1} \lambda(t) F(\dot{z(t)}, z(t), \theta, t) dt$$

$\lambda(t)$ is called a Lagrange multiplier[^n1]​. We know that $F=0$ so the additional term doesn’t change our gradient. That is
$$\tag{5} \frac{\mathrm{d} \psi}{ \mathrm{d} \theta} = \frac{\mathrm{d} L(z(t_1))}{ \mathrm{d} \theta}$$

So what did we gain? This: we can try to choose $\lambda(t)$ so that we can hopefully eliminate hard to compute derivatives in $\frac{\mathrm{d} L(z(t_1))}{ \mathrm{d} \theta}$, such as the computation graph Jacobian $\frac{\mathrm{d} z(t_1)}{\mathrm{d} \theta}$. In fact, let’s state that as a problem.
$$\tag{\text{PL}} \text{Choose } \lambda(t) \text{ to avoid } \\\\ \text{hard derivatives}$$

Solving $\text{(PL)}$ could help to solve $\text{(PG)}$, which helps for $\text{(PM)}$.

Simplify terms
--------------

Note: Rampant blatant argument and transpose suppression in what follows.😱

Let’s begin by investigating the integral at the end of (4).
$$
\begin{aligned}
\int_{t_0}^{t_1} \lambda(t) F dt &= \int_{t_0}^{t_1} \lambda(t) (\dot{z(t)} \medspace – \medspace f)dt \\\\
&= \int_{t_0}^{t_1} \lambda(t) \dot{z(t)} dt \enskip – \int_{t_0}^{t_1} \lambda(t) f dt \tag{6}
\end{aligned}
$$

Using [Integration by Parts](https://en.wikipedia.org/wiki/Integration_by_parts) for the first of the two terms, we see
$$\begin{aligned} \int_{t_0}^{t_1} \lambda(t) \dot{z(t)} dt &= \lambda(t)z(t) \big\vert_{t_0}^{t_1} – \int_{t_0}^{t_1} z(t) \dot{\lambda(t)} dt \\\\ &= \lambda(t_1)z(t_1) \medspace – \medspace \lambda(t_0)z_{t_0} \\\\ &- \int_{t_0}^{t_1} z(t) \dot{\lambda(t)} dt \end{aligned} \tag{7}$$

Using $(7)$ in $(6)$, we get
$$\begin{aligned} \int_{t_0}^{t_1} \lambda(t) F dt &= \lambda(t_1)z(t_1) \medspace – \medspace \lambda(t_0)z_{t_0} \\\\ &- \int_{t_0}^{t_1} (z \dot{\lambda} + \lambda f) dt \end{aligned}$$

Bringing in the derivative with respect to $\theta$, we see
$$\begin{aligned} \frac{\mathrm{d}}{\mathrm{d} \theta} \left\[ \int_{t_0}^{t_1} \lambda F dt \right\] &= \lambda(t_1)\frac{\mathrm{d} z(t_1)}{\mathrm{d} \theta} \medspace – \medspace \lambda(t_0)\overbrace{\cancel{\frac{\mathrm{d} z_{t_0} }{\mathrm{d} \theta}}^{\enskip 0}}^{z_{t_0} \text{ is input}} \\\\ &- \int_{t_0}^{t_1} (\frac{\mathrm{d} z}{\mathrm{d} \theta} \dot{\lambda} + \lambda \frac{\mathrm{d} f}{\mathrm{d} \theta} ) dt \end{aligned} \tag{8}$$

From the Chain rule we have
$$\frac{\mathrm{d} f}{\mathrm{d} \theta} = \frac{\partial f}{\partial \theta} + \frac{\partial f}{\partial z} \frac{\mathrm{d} z}{\mathrm{d} \theta}$$

So
$$\begin{aligned} \frac{\mathrm{d}}{\mathrm{d} \theta} \left\[ \int_{t_0}^{t_1} \lambda F dt \right\] &= \lambda(t_1)\frac{\mathrm{d} z(t_1) }{\mathrm{d} \theta} \\\\ &- \int_{t_0}^{t_1} (\dot{\lambda} + \lambda \frac{\partial f}{\partial z} ) \frac{\mathrm{d} z}{\mathrm{d} \theta}dt \\\\ &- \int_{t_0}^{t_1} \lambda \frac{\partial f}{\partial \theta} dt \end{aligned} \tag{9}$$

We can now go all the way back to $(4)$ and differentiate with respect to $\theta$. Invoking $(5)$ and the Chain Rule, we get
$$\frac{\mathrm{d} L}{ \mathrm{d} \theta} = \frac{ \partial L}{ \partial z(t_1)}\frac{\mathrm{d} z(t_1)}{ \mathrm{d} \theta} \enskip – \enskip \frac{\mathrm{d}}{\mathrm{d} \theta} \left\[ \int_{t_0}^{t_1} \lambda F dt \right\]$$

Finally, let’s pull in $(9)$ and collect terms.
$$\begin{aligned} \frac{\mathrm{d} L}{ \mathrm{d} \theta} &= \left\[ \frac{ \partial L}{ \partial z(t_1)} – \lambda(t_1) \right\] \frac{\mathrm{d} z(t_1) }{\mathrm{d} \theta} \\\\ &+ \int_{t_0}^{t_1} (\dot{\lambda(t)} + \lambda(t) \frac{\partial f}{\partial z} ) \frac{\mathrm{d} z(t)}{\mathrm{d} \theta}dt \\\\ &+ \int_{t_0}^{t_1} \lambda(t) \frac{\partial f}{\partial \theta} dt \end{aligned} \tag{10}$$

Solve PL, PG, PM with Good Lagrange Multiplier
----------------------------------------------

We can now stop and think about $\text{(PL)}$ – avoid hard derivatives. Let’s list the derivatives in $(10)$.

| Derivative                                         | Description                                                                                                               | Ease    |
|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|---------|
| \\(\frac{ \partial L}{ \partial z(t_1)}\\)         | Gradient of loss with respect to output.                                                                                  | Easy    |
| \\(\frac{\mathrm{d} z(t_1) }{\mathrm{d} \theta}\\) | Jacobian of output with respect to params.                                                                                | Not easy|
| \\(\dot{\lambda(t)}\\)                             | Derivative of lambda, a vector, with respect to time.                                                                     | Easy    |
| \\(\lambda(t) \frac{\partial f}{\partial z}\\)     | vector-Jacobian Product. Can compute with [reverse mode autodiff](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#reverse-mode-differentiation) [without explicitly constructing Jacobian](/how-is-the-vector-jacobian-product-invoked-in-neural-odes/) \\(\frac{\partial f}{\partial z}\\). | Easy    |
| \\(\frac{\mathrm{d} z(t)}{\mathrm{d} \theta}\\)    | Jacobian of arbitrary layer with respect to params.                                                                       | Not easy|
| \\(\lambda(t) \frac{\partial f}{\partial \theta}\\)| vector-Jacobian Product. Can compute with [reverse mode autodiff](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#reverse-mode-differentiation) [without explicitly constructing](/how-is-the-vector-jacobian-product-invoked-in-neural-odes/) Jacobian\\(\frac{\partial f}{\partial \theta}\\). | Easy    |


So we see that $(10)$ requires calculating $\frac{\mathrm{d} z(t)}{\mathrm{d} \theta}$ at the very last point in time $t_1$ (output layer) and possibly any general point in time $t$ (arbitrary hidden layer). That is exactly what we’re trying to avoid!

Can we get rid of $\frac{\mathrm{d} z(t)}{\mathrm{d} \theta}$ from $(10)$ by making a judicious choice of $\lambda(t)$?🤔

What if we define $\lambda(t)$ to be such that it satisfies the following ODE that runs backwards in time, starting at $t_1$ and ending at $t_0$
$$\boxed{\begin{aligned} &\dot{\lambda(t)} = -\lambda(t)\frac{\partial f}{\partial z} \enskip \text{s.t.} \enskip \lambda(t_1) = \frac{ \partial L}{ \partial z(t_1)} \\\\ &\text{giving} \\\\ &\lambda(t_0) = \lambda(t_1) – \int_{t_1}^{t_0} \lambda(t) \frac{\partial f}{\partial z} dt \end{aligned}} \tag{11}$$

Like magic, the first two terms in $(10)$ – the ones containing $\frac{\mathrm{d} z}{\mathrm{d} \theta}$ – will simply zero out!!!😃🥳

$(10)$ then simplifies to
$$\boxed{\frac{\mathrm{d} L}{ \mathrm{d} \theta} = -\int_{t_1}^{t_0} \lambda(t) \frac{\partial f}{\partial \theta} \tag{12} dt}$$

where I just flipped the integration limits​[^n2]​.

Notice $\frac{\mathrm{d} L}{ \mathrm{d} \theta}$ is precisely the loss gradient we’re after in $\text{(PG)}$!

But you’ll remember from $\text{(PG)}$ that just getting the gradient is not enough – we want the gradient computation to be memory efficient. Well, notice that $(12)$ is a new ODE system that does not require us to preserve activations from the forward pass. So our new method trades off computation for memory – in fact the memory requirement for this gradient calculation is $\text{O}(1)$ with respect to the number of layers!

Once we have the gradient, we can readily use it in gradient descent to (locally) solve $\text{(PM)}$!

Summary of Steps
----------------

In summary, here are the steps in one iteration of solving $\text{(PM)}$ with gradient descent (for batch of one):

1.  Forward pass: Solve the ODE $(1)$-$(2)$ from time $t_0$ to $t_1$ and get the output $z(t_1)$.
2.  Loss calculation: Calculate $L(z(t_1))$.
3.  Backward pass: Solve ODEs $(11)$ and $(12)$ from reverse time $t_1$ to $t_0$ to get the gradient of the loss $\frac{\mathrm{d} L(z(t_1))}{ \mathrm{d} \theta}$.
4.  Use the gradient to update the network parameters $\theta$.

What is the adjoint?
--------------------

The $\lambda(t)$ we defined in $(11)$ is called the adjoint. I think one reason for the name is that adjoint refers to the transpose in linear algebra, and $\lambda(t)$ is a row vector or equivalently a transposed column vector. $(11)$ itself is the adjoint system.

While the paper derives these equations using an alternate proof that proceeds by defining adjoint and then using Chain Rule, we have _derived_ the expressions for the adjoint and the adjoint equations here using the method of Lagrange multipliers. For kicks.

If we compare $(11)$, $(12)$ to the equations $(46)$, $(51)$ in the original Neural ODEs paper, we see that all the definitions of the adjoint, adjoint system and the gradient of the loss with respect to the parameters are in agreement. In fact $\lambda(t)$ corresponds to the quantity $a(t)$ defined in the paper.. woot!!🎉​[^n3]

Comments ported from Wordpress on “Deriving the Adjoint Equation for Neural ODEs using Lagrange Multipliers”
------------------------------------------------------------------------------------------------------------

> **Kacper** said on `2020-02-05 09:47:43`:

Hi!

Thank you for your post. I will jump derivation more in-depth (as I'm a newbie in these topics). What I lack at first glance, is an explanation for a dot notation above a letter (I derived from the table that it is a derivative - I know it's standard notation in physics but rather uncommon in ML community). I think it would be worth to mention what $\dot{z}$ stands for in the beginning.

Best regards


> **Vaibhav Patel** said on `2020-02-05 10:24:34`:

Hey, thanks for the feedback! Really appreciate it!

I agree, I tried unsuccessfully to not be all over the place with my notations. Part of it has to do with not being adept at knowing how to fit equations on smaller devices.. I just started blogging and I thought it would all just.. flow.. or something.

For starters, I'll definitely try to add a line above explaining that the dot notation signifies derivative with respect to time.


> **Peter Gall** said on `2020-02-19 22:27:01`:

Hey, thanks a lot for your writeup.

I am a little confused about how you would actually compute the vector-Jacobian product with autodiff without explicitly constructing the Jacobian.

Do you have mock code that shows this? In e.g. pytorch?

> **Vaibhav Patel** said on `2020-02-21 10:14:55`:

Hey, that's a great question! (I've edited my previous reply here to be hopefully more lucid. I also wrote <a href="https://vaipatel.com/how-is-the-vector-jacobian-product-invoked-in-neural-odes/" rel="noopener noreferrer" target="_blank">separate post</a> to address this question and provided some code snippets there as well. Hope you find it useful!)

The vector-Jacobian product can be calculated using an autograd library like <a href="https://github.com/HIPS/autograd" target="_blank" rel="noopener noreferrer nofollow ugc">HIPS/autograd</a> or <a href="https://pytorch.org/docs/stable/autograd.html" target="_blank" rel="noopener noreferrer nofollow ugc">pytorch.autograd</a>. In such libraries the vjp is calculated using <a href="https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#reverse-mode-differentiation" target="_blank" rel="noopener noreferrer nofollow ugc">reverse-mode autodiff</a>. The idea is that instead of instantiating full Jacobians, these libraries use precomputed simplified expressions of vjp's for all functions one might encounter in deep learning. The computational cost of these simplified expressions is of the same order as their original functions owing to the sparsity of the Jacobians and the commonality in expressions of the functions and their derivatives. This is much cheaper than instantiating a full Jacobian matrix and then left multiplying it by a row vector.

Here are <a href="https://github.com/rtqichen/torchdiffeq/blob/55fa6af17afc51c728c67b856433d49143229393/torchdiffeq/_impl/adjoint.py#L41-L44" target="_blank" rel="noopener noreferrer nofollow ugc">the exact lines (41-44) calling pytorch's vjp</a> in the <a href="https://github.com/rtqichen/torchdiffeq" target="_blank" rel="noopener noreferrer nofollow ugc">torchdiffeq</a> package by the authors.

```py
vjp_t, *vjp_y_and_params = torch.autograd.grad(
    func_eval, (t,) + y + f_params,
    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
)
```

This snippet is requesting the 3 efficient vector-Jacobian products. The left row vector in each case is the adjoint vector <code>adj_y</code>. The Jacobians are the derivative of <code>func_eval</code> with respect to i) the time at timepoint <code>t</code>,  ii) the state <code>y</code> at time point t, and iii) the neural net params <code>f_params</code>.

Actually, the above autodiff way of calculating the vjp is just the backpropagation of the gradient <code>adj_y</code> through <code>f</code>. The crucial point is the neural net <code>f</code> is not our entire computation graph from the ODE solve - it's an evolution function that just takes the state <code>y</code> to the "next" state in time.

Cheers!


> **Pere Diaz Lozano** said on `2020-10-14 11:30:21`:

Good afternoon,

Would you use the same techique to derive the expression to compute the gradient of the loss with respect to the initial time t0?

I've been trying to do it but can not figure it out.

Thank you very much


> **Vaibhav Patel** said on `2020-10-25 14:59:53`:

Hi there, yes the gradient with respect to the initial time $t_0$ can also be determined using the same technique. To frame an initial value problem for getting $\frac{\mathrm{d} L}{\mathrm{d} t_0}$, we need
1. The initial value $\frac{\mathrm{d} L}{\mathrm{d} t_n}$
2. And ODE for $\frac{\mathrm{d}}{\mathrm{d}t}(\frac{\mathrm{d} L}{\mathrm{d} t})$

For the initial value, we have $\frac{\mathrm{d} L}{\mathrm{d} t_n} = \frac{\mathrm{d} L}{\mathrm{d} z} \frac{\mathrm{d} z}{\mathrm{d} t_n} = \lambda(t_n) f(z(t_n), \theta, t_n)$

For the ODE, we have $\frac{\mathrm{d}}{\mathrm{d}t}(\frac{\mathrm{d} L}{\mathrm{d} t}) = \frac{\mathrm{d}}{\mathrm{d}t}(\frac{\mathrm{d} (\lambda f)}{\mathrm{d} t}) = \dot{\lambda} f + \lambda \frac{\mathrm{d} f}{\mathrm{d} t} = -\lambda \frac{\partial f}{\partial z} f + \lambda \frac{\partial f}{\partial z}\frac{\mathrm{d} z}{\mathrm{d} t} + \lambda \frac{\partial f}{\partial t} = -\lambda \frac{\partial f}{\partial z} f + \lambda \frac{\partial f}{\partial z} f + \lambda \frac{\partial f}{\partial t} = \lambda \frac{\partial f}{\partial t}$

So $\frac{\mathrm{d} L}{\mathrm{d} t_0} = \frac{\mathrm{d} L}{\mathrm{d} t_n} + \int_{t_1}^{t_0} {\lambda(t) \frac{\partial f}{\partial t} dt}$.

In the paper, it looks like they instead evaluate $\frac{\mathrm{d} L}{\mathrm{d} t_0} = -( -\frac{\mathrm{d} L}{\mathrm{d} t_n} - \int_{t_1}^{t_0} {\lambda(t) \frac{\partial f}{\partial t} dt})$ to be able to get the dynamics for $\lambda(t)$, $\frac{\mathrm{d} L}{\mathrm{d} \theta}$ and $\frac{\mathrm{d} L}{\mathrm{d} t}$ by performing a single vector-Jacobian product between the adjoint $\lambda(t)$ and the Jacobian of $f(z, \theta, t)$.


> **Rishant** said on `2022-05-20 23:05:57`:

In the derivation, to obtain eq (8), λ(t) is considered independent of θ.
Then we set a specific form of λ(t) in (11).
Does it follow from (11) that indeed λ(t) as chosen is independent of θ?

> **Vaibhav Patel** said on `2024-04-06 14:01:26`:

Hi, apologies for the delayed reply. Great question. Loosely speaking, I comforted myself with the idea that the adjoint state $\lambda(t)$ was somewhat analogous to $z(t)$, in that they both may have a dependence on $\theta$ as "parameters" but not inputs.

For instance if $\dot{z(t)} = f(z(t), \theta, t) = \theta z(t)$, then $z(t) = z_{t_0} e^{\theta z(t)}$, and so $z(t)$ is more like $z_{\theta}(t)$.

Analogously, from (11) we have $\dot{\lambda(t)} = -\lambda(t) \frac{\partial f}{\partial z} = -\theta \lambda(t)$. So $\lambda(t) = \lambda(t_0) e^{-\theta t}$. There is definitely a parameter dependence on $\theta$, but I take it to similarly mean that we should write $\lambda_{\theta}(t)$.

Further given the independence of $\theta$ from $t$, I think (11) is definitely "safer" because $\frac{\mathrm{d} \lambda}{\mathrm{d} t} = \frac{\partial \lambda}{\partial t} + \frac{\partial \lambda}{\partial \theta}\frac{\mathrm{d} \theta}{\mathrm{d} t} = \frac{\partial \lambda}{\partial t}$.

Anyway, I concede this isn't very rigorous. Seemingly, https://arxiv.org/html/2402.15141v1 has a proof that clarifies the dependence of the adjoint state on the parameters.

> **Soumya** said on `2022-11-01 20:12:28`:

Hi,  

Thank you very much for the detailed explanation. Could you please explain in the neural ode paper equation (45) why the $a_\theta(t_n) =0$.

> **Vaibhav Patel** said on `2024-04-06 16:28:17`:

Hi there, sorry about the delay. It was admittedly a bit sneaky on my part - I actually swapped the limits of the integration and then stuck a minus sign in front. The form was influenced by the Neural ODEs paper where the solve the ODEs for the adjoint and $\frac{\mathrm{d} L}{\mathrm{d} \theta}$ (which (11) and (12) should correspond to) in the same ODE pass from $t_1$ to $t_0$.

> **Jack** said on `2023-08-13 04:37:54`:

Thanks for your writeup and I wonder if there's a typo: should the upper bound and the lower bound in your integral be t_0 and t_n?


> **Vaibhav Patel** said on `2024-03-30 20:37:23`:

Hi there, apologies for the very late reply. Great question. My interpretation is that since $a_\theta(t_n) = \frac{\mathrm{d} L}{\mathrm{d} \theta(t_n)}$, by setting $a_\theta(t_n) = 0$ and flowing the adjoint system backward, we're expressing our desire to find the params that will land us in a local minimum of $L$ when the system is run in forward time. Basically, doing stochastic gradient descent to a local minimum of the loss for the current batch of training samples.


References
----------

[^n1]: Actually I believe in optimal control theory it is more appropriately called a costate variable.
[^n2]: A reason is that this way, all the ODEs associated with our gradient calculation run backwards. This consistency can help make the calls to our ODE solver more concise.
[^n3]: I think to say that 𝜆 and a are the same, we need 𝜆 to be uniformly Lipschitz continuous, after which we can invoke Picard’s uniqueness theorem.
[^1]: Chen TQ, Rubanova Y, Bettencourt J, Duvenaud DK. Neural ordinary differential equations. In: _Advances in Neural Information Processing Systems_. ; 2018:6571–6583.
[^2]: Pontryagin LS, Mishchenko E, Boltyanskii V, Gamkrelidze R. The mathematical theory of optimal processes. 1962.
[^3]: Bradley AM. _PDE-Constrained Optimization and the Adjoint Method_. Technical Report. Stanford University. https://cs. stanford. edu/ ambrad …; 2013.
[^​4]: Cao Y, Li S, Petzold L, Serban R. Adjoint Sensitivity Analysis for Differential-Algebraic Equations: The Adjoint DAE System and Its Numerical Solution. _SIAM Journal on Scientific Computing_. 2003;24:1076-1089. doi:[10.1137/S1064827501380630](https://doi.org/10.1137/S1064827501380630)
[^​5​]: Biegler LT. Optimization of Differential-Algebraic Equation Systems. _Chemical Engineering Department Carnegie Mellon University Pittsburgh, http://dynopt cheme cmu edu_. 2000.