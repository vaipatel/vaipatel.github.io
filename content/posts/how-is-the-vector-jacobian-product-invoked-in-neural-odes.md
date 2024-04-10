+++
title = "How is the vector-Jacobian product invoked in Neural ODEs"
date = "2020-02-21T10:09:59"
tags = ["Machine Learning"]
keywords = ["Machine Learning"]
slug = "how-is-the-vector-jacobian-product-invoked-in-neural-odes"
draft = false
showtoc = true
tocopen = false
+++

This post just tries to explicate the claim in [Deriving the Adjoint Equation for Neural ODEs Using Lagrange Multipliers](/posts/deriving-the-adjoint-equation-for-neural-odes-using-lagrange-multipliers/) that the vector-Jacobian product $\lambda^\intercal \frac{\partial f}{\partial z}$ can be calculated efficiently without explicitly constructing the Jacobian $\frac{\partial f}{\partial z}$. The claim is made in the [Solving PL, PG, PM with Good Lagrange Multiplier](/posts/deriving-the-adjoint-equation-for-neural-odes-using-lagrange-multipliers/#solve-pl-pg-pm-with-good-lagrange-multiplier) section.

This post is inspired by [a question asked about this topic in the comments post there](/posts/deriving-the-adjoint-equation-for-neural-odes-using-lagrange-multipliers/#:~:text=Do%20you%20have%20mock%20code%20that%20shows%20this%3F%20In%20e.g.%20pytorch%3F).

In what follows, the variable $y$ will take the place of $z$ from the earlier post.

torchdiffeq uses torch.autograd.grad’s vJp magic
------------------------------------------------

To begin let’s see how [torchdiffeq](https://github.com/rtqichen/torchdiffeq), the Neural ODEs implementation from the original authors, calls [pytorch AutoDiff’s](https://pytorch.org/docs/stable/autograd.html#module-torch.autograd) [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad) function to calculate the vector-Jacobian product. The relevant chunk is in [lines 37-46](https://github.com/rtqichen/torchdiffeq/blob/55fa6af17afc51c728c67b856433d49143229393/torchdiffeq/_impl/adjoint.py#L37-L46). Specifically, the call to [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad) is made between [lines 41-44](https://github.com/rtqichen/torchdiffeq/blob/55fa6af17afc51c728c67b856433d49143229393/torchdiffeq/_impl/adjoint.py#L41-L44).


```py
with torch.set_grad_enabled(True):
    t = t.to(y[0].device).detach().requires_grad_(True)
    y = tuple(y_.detach().requires_grad_(True) for y_ in y)
    func_eval = func(t, y)
    vjp_t, *vjp_y_and_params = torch.autograd.grad(
        func_eval, (t,) + y + f_params,
        tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
    )
vjp_y = vjp_y_and_params[:n_tensors]
vjp_params = vjp_y_and_params[n_tensors:]
```

Here’s a table to clarify the arguments and return values of the [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad) function call. (For simplicity, I assume the loss depends on just the terminal timepoint $t_1$ and that the the batch size is $1$. Further, I ignore $f$'s other inputs $t, \theta$.)


| Code | Math | Description |
| --- | --- | --- |
| `func_eval` | $f(t_1, y(t_1), \theta)$ | (arg) The output whose derivative you want. |
| `y` | $y(t_1)$ | (arg) The state w.r.t to which you want output’s derivative. |
| `adj_y` | $\lambda(t_1)^\intercal$ | (arg) The free choice adjoint vector/lagrange multiplier. |
| `vjp_y` | $\lambda(t_1)^\intercal \frac{\partial f}{\partial y(t_1)}$ | (ret) The vector-Jacobian product. |

In this case, `torch.autograd.grad(func_eval, y, adj_y)` does all the magic under the hood to compute $\lambda(t_1)^{\intercal} \frac{\partial f}{\partial y(t_1)}$ without constructing the Jacobian $\frac{\partial f}{\partial y(t_1)}$. The same principle applies to calculating $\lambda(t_1)^\intercal \frac{\partial f}{\partial \theta}$.

How does [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad) compute the vector-Jacobian product without constructing the Jacobian? Could it really be.. magic😮?

torch.autograd.grad’s vJp is not really magic
---------------------------------------------

No, [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad)'s vector-Jacobian product calculation is not really magic.😶

To be sure, the underlying idea called reverse mode (as opposed to forward mode) automatic differentiation is ingenious and elegant. But it helps (me) to sometimes see it as just another case of optimization-by-upfront-work. Here is how we would implement reverse mode autodiff based vector-Jacobian products.

1. Precompute the simplified expressions of $\text{vjp}_{(f, y)}(v) = v^\intercal \frac{\partial f(y)}{\partial y}$ for a whole bunch of $f(y)$'s​ [^n1]​.
2. Simply invoke $\text{vjp}_{(f, y)}(\lambda)$ to calculate $\lambda^\intercal \frac{\partial f(y)}{\partial y}$.

That’s really all there is to the reverse mode autodiff way of calculating vector-Jacobian products without Jacobians. Specifically, in step 1, the simplified expression $\text{vjp}_{(f, y)}(v)$ will help exploit

* the sparsity of the Jacobian and
* the commonality in the expression for a function and its derivative

to avoid/reuse calculations so that we never have to hold the entire Jacobian in memory. We can just calculate the product very easily from $f, y$ and the transposed vector $v$.

In fact we only need atmost **three times the cost in FLOPS needed for calculating the original function $f$, independent of the dimensionality of the input** [^​1]​.😲

Let’s see the autodiff idea for vector-Jacobian products with a few examples.

Example 1 of vJp without Jacobian: $sin(y)$
-------------------------------------------

In this example we’ll explore how autodiff vector-Jacobian product implementation exploits Jacobian sparsity.

It is a bit contrived but imagine the state is $y(t)=[y_1(t), y_2(t), y_3(t)]$ and the NN is $f(t, y(t), \theta) = \sin(y(t)) = [\sin(y_1(t)), \sin(y_2(t)), \sin(y_3(t))]$. Since both $f$ and $y$ are 3d, the full Jacobian $\frac{\partial f}{\partial y}$ is a $3×3$ matrix as follows.

$$\begin{aligned} \frac{\partial f}{\partial y} &= \begin{bmatrix} \frac{\partial f_1}{\partial y_1} & \frac{\partial f_1}{\partial y_2} & \frac{\partial f_1}{\partial y_3} \\\\ \frac{\partial f_2}{\partial y_1} & \frac{\partial f_2}{\partial y_2} & \frac{\partial f_2}{\partial y_3} \\\\ \frac{\partial f_3}{\partial y_1} & \frac{\partial f_3}{\partial y_2} & \frac{\partial f_3}{\partial y_3} \end{bmatrix} = \begin{bmatrix} \cos(y_1) & 0 & 0 \\\\ 0 & \cos(y_2) & 0 \\\\ 0 & 0 & \cos(y_3) \end{bmatrix} \\\\ & = \cos(y) \end{aligned}$$


Thus the vector-Jacobian product in this case would be $$\begin{aligned} \lambda^\intercal \frac{\partial f}{\partial y} & = \begin{bmatrix} \lambda\_1 \cdot \cos(y\_1) & \lambda\_2 \cdot \cos(y\_2) & \lambda\_3 \cdot \cos(y\_2) \end{bmatrix} \\\\ &= (\lambda \odot \cos(y))^\intercal \end{aligned} \tag{1}$$


where $\odot$ is the elementwise multiplication operator.


So we really don’t need to store the 3×3 Jacobian matrix $\frac{\partial f}{\partial y}$ to calculate the vector-Jacobian product $\lambda^\intercal \frac{\partial f}{\partial y}$. Instead we can make a vector-Jacobian product **routine** that just exploits $\text{(1)}$ to calculate the product.


```py
def vjp_sin(f, y):
  """For f = sin(y), returns v.T * df/dy without making df/dy."""
  return lambda v: np.multiply(v, np.cos(y))

# Suppose
# 1. y is the input vector
# 2. adj_y is the adjoint vector

# forward pass
func_eval = np.sin(y) # (store func_eval for later)

# vjp from reverse mode autodiff
vjp_result = vjp_sin(func_eval, y)(adj_y)
```

Notice that `vjp_sin` never keeps around the full jacobian $\frac{\partial f}{\partial y}$. Also notice that the `lambda` function on line 3 only takes about twice as many flops as the original $\sin$ func – about the same for the `cos` and the `multiply`.


Example 2 of vJp without Jacobian: $sin(Wy)$
--------------------------------------------

In this example we’ll explore how autodiff vector-Jacobian product implementation exploits commonality in the expressions for a function and its derivative.

Suppose $f(z) = \sin(z)$, where $z=Wy$ for some weight matrix $W$. We want $v^\intercal \frac{\partial f}{\partial W}$. We’ll assume that $y$ is 3d, $W$ is $2 \times 3$ and $z, f(z), v$ are 2d.


From the chain rule we have $$v^\intercal \frac{\partial f}{\partial W} = v^\intercal \frac{\partial f}{\partial z} \frac{\partial z}{\partial W} = \left( v^\intercal \frac{\partial f}{\partial z} \right) \frac{\partial z}{\partial W}$$


First, notice that the vector-Jacobian product in the brackets can be easily evaluated using the `vjp_sin` implementation of $\text{(1)}$ from the preceding section. Thus $$v^\intercal \frac{\partial f}{\partial W} = (v \odot \cos(z))^\intercal \frac{\partial z}{\partial W}$$


Now the Jacobian $\frac{\partial z}{\partial W}$ is not exactly sparse. But it is easy (though laborious) to show that for $z = Wy$, the vector-Jacobian product ${v^\prime}^\intercal \frac{\partial z}{\partial W}$ is just the outer product ${v^\prime} \otimes y$. See [^2]​. Thus $$v^\intercal \frac{\partial f}{\partial W} = (v \odot \cos(z)) \otimes y$$


Here’s how we could code this up.

```py
def vjp_sin(f, z):
  """Just copied from the preceding section. Remember f = sin(z)."""
  return lambda v: np.multiply(v, np.cos(z))

def vjp_matmul_W(z, W, y):
  """This is the vjp for v.T dz/dW = v.T d(Wy)/W"""
  return lambda v: np.outer(v, y) # No need to transpose v or y. Numpy smart.

# Suppose 
# 1. y is the input vector
# 2. W is a weight matrix
# 3. adj_y is the adjoint vector

# forward pass
z = np.dot(W, y) # (store z for later)
func_eval = np.sin(z) # (store func_eval for later)

# vjp from reverse mode autodiff
vjp_intermediate = vjp_sin(func_eval, z)(adj_y)
vjp_result = vjp_matmul_W(z, W, y)(vjp_intermediate)
```

Again hopefully it is clear that no where are we instantiating full Jacobians.


### Notes for making an actual reverse mode autodiff library


In an actual library, we would obviously want to make our calls a lot more natural and consistent. So we would do a few things like


1. Pair functions to their vector-Jacobian product implementations in a dictionary like data structure so as to execute our reverse pass in an automated fashion with a simple call to a high level function like `some_loss.reverse()`.
2. Extend the above logic to many other types of functions like binary functions etc.


How reverse mode vJp helps for backprop in DL
---------------------------------------------


Hopefully example 2 above is already starting to reveal why reverse mode autodiff is chosen for backprop implementations in deep learning libraries.


Reverse mode autodiff helps efficiently calculate $\frac{\partial f}{\partial y}$ when $f$ is a deep composition like $$f(y) = L(f\_l(f\_{l-1}(..(f\_1(y)))))$$ **and the outermost loss function L outputs a scalar**.


By the chain rule we’d have $$\begin{aligned}\frac{\partial f}{\partial y} &= \frac{\partial L}{\partial f\_l} \frac{\partial f\_l}{\partial f\_{l-1}} \frac{\partial f\_{l-1}}{\partial f\_{l-2}} … \frac{\partial f\_{2}}{\partial f\_{1}} \\\\ &= \left(\left(\left( \frac{\partial L}{\partial f\_l} \frac{\partial f\_l}{\partial f\_{l-1}} \right) \frac{\partial f\_{l-1}}{\partial f\_{l-2}}\right) …\right) \frac{\partial f\_{2}}{\partial f\_{1}} \tag{2} \end{aligned}$$


where we get the second expression by invoking the associativity of matrix multiplication.


Now notice that because $L$ is a scalar, the Jacobian $\frac{\partial L}{\partial f\_l}$ is just a row vector! Let’s call the transpose of this row vector $g$. Then $\text{(2)}$ becomes $$\begin{aligned}\frac{\partial f}{\partial y} &= \left(\left(\left( g^\intercal \frac{\partial f\_l}{\partial f\_{l-1}} \right) \frac{\partial f\_{l-1}}{\partial f\_{l-2}}\right) …\right) \frac{\partial f\_{2}}{\partial f\_{1}} \tag{3} \end{aligned}$$


Starting from the innermost parenthesis, we can just invoke the appropriate vector-Jacobian product function and move out to the next enclosing parenthesis. At that iteration we’ll find ourselves with yet another task of left-multiplying a Jacobian by a row vector, which we can once again compute efficiently using the appropriate vector-Jacobian product function. So on and so forth till we’ve successfully finished all products in the chain.


Hopefully the previous section and $\text{(3)}$ make the advantage and drawback of backprop abundantly clear.


* The advantage is that we can efficiently calculate the gradient of a scalar loss with respect to a very high-dimensional input (or parameter tensor).
* The drawback is that we need to store in memory all the function evaluations from our forward pass.


Back to Neural ODEs
-------------------


Circling back to Neural ODEs, remember that we invoke reverse mode autodiff to calculate the vector-Jacobian product between the adjoint vector $\lambda(t\_1)$ and the Jacobian $\frac{\partial f}{\partial y(t\_1)}$ of the evolution function neural network $f(y)$.


But from $\text{(3)}$ it looks like this would set off a chain of vector-Jacobian products, essentially giving us backprop with the only difference being that the vector $g$ is replaced by the adjoint vector $\lambda(t\_1)$.


What’s going on here? Don’t Neural ODEs not use backprop? 🤔


Well, actually it would be correct that this would look very much like backprop. The crucial difference is that we would not be backpropagating gradients from the loss all the way through the potentially huge computation graph created by the ODE solver steps.


Instead we would only be “backpropagating” the adjoint vector​ [^n2]​ from $f(y(t\_1))$ to $y(t\_1)$ which is a very tiny graph in comparison the entire ODE solver’s computation graph.😀


Hence the claim in [Deriving the Adjoint Equation for Neural ODEs Using Lagrange Multipliers](/posts/deriving-the-adjoint-equation-for-neural-odes-using-lagrange-multipliers/) that the vector-Jacobian product $\lambda^\intercal \frac{\partial f}{\partial z}$ can be calculated efficiently without explicitly constructing the Jacobian $\frac{\partial f}{\partial z}$.

References
----------
[^n1]: ​For simplicity, I’m considering only unary functions.
[^n2]: ​I believe the actual term here is pulling back.
[^​1]: Griewank A. A mathematical view of automatic differentiation. *Acta Numerica*. 2003;12:321–398.
[^2]: Clark K. Computing Neural Network Gradients. Stanford CS224n: Natural Language Processing with Deep Learning. <https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf>. Published December 20, 2019. Accessed February 25, 2020.

