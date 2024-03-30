+++
title = "Semilandmarks: Abridged"
date = "2020-01-07T06:51:08"
tags = ["Geometric Morphometrics"]
keywords = ["Geometric Morphometrics"]
slug = "semilandmarks-abridged"
draft = false
showtoc = true
tocopen = false
+++

In Geometric Morphometrics the study of the shape of a species begins at the identification of homologous landmarks across specimens in a dataset.


Think the tip of the nose, or the corner of the eye.


The patterns in landmark variations that survive the locations and orientations of the specimens are taken to represent true shape variations.


But, as explained in [Semilandmarks in Three Dimensions](https://www.researchgate.net/profile/Philipp_Gunz/publication/226696996_Semilandmarks_in_Three_Dimensions/links/09e41509bfbe573b17000000.pdf) [^1], homologous regions don’t always neatly fit into discrete landmarks.


Landmarks are not enough
------------------------


There are many cases in nature where the structures of homology are contiguous regions. 


For example, there are curving structures like the outline of a butterfly’s wings, or even surfaces, like the entire human "brain case" (Figure 1).


![](/images/semilmks_neurocranial_anim.gif)**Figure 1**. The human neurocranial vault. While there is a dearth of discrete landmarks, the entire surface can be readily apprehended as homologous.  
Polygon data is from BodyParts3D, CC BY-SA 2.1 jp, https://commons.wikimedia.org/w/index.php?curid=37589706

There may not be any single identifiable landmark on these contiguous regions. And yet the entire machinery of Geometric Morphometrics relies on landmarks sets.


So how to relate contiguous regions across specimens using a technique that relies on corresponding landmarks?


Perhaps we could sample these regions using some sampling strategy?🤨


Equidistant Landmarks are wrong
-------------------------------


A naive strategy to relating curves/surfaces would be to sample them with equidistant landmarks and then relate those landmarks.


Unfortunately, this could lead to spurious shape variations.


![](/images/semilmks_in_3d_rect.png?w=1024)**Figure 2**. Notice how point 24 occurs at the corner of specimens **a** and **c** but not **b**. From [^1].

For example, imagine regularly sampling the outlines of rectangular specimens. The corners of two specimens could very well become wrongly misaligned (Figure 2).


![](/images/semilmks_in_3d_rectwarp.png?w=1024)**Figure 3**. TPS deformation grids from a) **2a** to **2b** and b) **2a** to **2c** reveal the flaw in equidistant sampling. Specifically, notice the spurious increased bending from **2a** to **2b** owing to misaligned corner landmarks. From [^1].

Of course, the corners are likely truly homologous, and so shape differences from misaligned corners would be an artifact of our sampling (Figure 3).


But there is a baby in the bath water. What if we grant the equidistant landmarks a chance to "slide" about the curve/surface so as to minimize the artificial differences?🤔


Semilandmarks, in principle
---------------------------


Semilandmarks are landmark samples on curves/surfaces that get some leeway to "slide" into their final positions.


Take the illustration in Figure 4. Here $c_1$, $c_2$ are curves, while $s_1$ is a surface patch with a polar grid.


![](/images/semilmks_illus_terrible.png)**Figure 4**. Ideal sliding step for a member from a made-up species. The curve semilandmark $a$ slides to $a^{\prime}$ along $c_1$. Similarly, $b$ slides on surface $s_1$ to $b^{\prime}$.

An *ideal* "sliding step" would let $a$ slide along curve $c_1$ and settle into $a^{\prime}$. Similarly, $b$ would slide on the surface patch $s_1$ and settle into $b^{\prime}$.


Now, what objective should the sliding step be trying to minimize? Per the authors in [^1], the objective should have the following desiderata.


1. Should minimize the visual expression of distortion caused by inadequacies of our sampling rather than the "actual" data.
2. Should not interfere with routines in [procrustes analysis](https://en.wikipedia.org/wiki/Procrustes_analysis).


Lucky for us, the **bending energy** of the [thin-plate spline](http://mathworld.wolfram.com/ThinPlateSpline.html) (TPS) [^2] does both😀. See below for a [rough expo based on elasticity](#why_minimize_bending_energy).


So, our sliding step should slide specimen semilandmarks so as to minimize the bending energy of the TPS warp from our specimen’s landmark set to a reference set.


That’s the principle.


### Not easy to parametrically slide on curves, meshes


What do we mean by "slide" the semilandmarks? And what specifically are we solving for?


Well, to move a point along a curve or surface, we would need to solve for its initial velocity, which would be tangent to the respective curve/surface at that point, and then push it along a [geodesic](https://en.wikipedia.org/wiki/Geodesic) emanating from it with that initial velocity.


Finding the best geodesic for the task would require us to parameterize the variation in geodesics around each semilandmark.


**This isn’t an area I’m deeply familiar with**, but I think this is onerous for all but the simplest cases. We may be able to leverage concepts from differential geometry [^3] [^n1], but they assume [smooth Riemannian manifolds](https://en.wikipedia.org/wiki/Riemannian_manifold) with a nice parametric representations while, in fact, in the overwhelming majority of cases our specimen model would be a discrete mesh.


Which means very likely we’d have to resort some kind of black-box non-linear optimization to get the tangent velocities at all the semilandmarks. 😐 


Could there be a way to make the above optimization problem simpler?


### Linearize the sliding


Yes! 😃 


Say we relax the stringent condition that the semilandmarks remain confined to their curve/surface and instead let the semilandmarks move away from the initial positions in their tangent hyperplanes – 1d line for curve semilandmarks and 2d plane for surface semilandmarks.


It turns out that this simpler problem is a kind of [Generalized Least Squares](https://en.wikipedia.org/wiki/Generalized_least_squares) (GLS) problem of the form  $${\underset {T}{\operatorname {argmin} }} (\mathbf{\Upsilon^0} + \mathbf{U}T)^{\intercal} \mathbf{L_k^{-1}} (\mathbf{\Upsilon^0} + \mathbf{U}T)$$  where $\mathbf{\Upsilon^0}$ encodes the initial positions of all the landmarks including semilandmarks, $\mathbf{U}$ encodes the tangent directions, $T$ are the tangent weights we want to solve for, and $\mathbf{L_k^{-1}}$ is a block-diagonal version of the bending energy matrix $L_k^{-1}$.


The bending energy matrix $L_k^{-1}$ is a submatrix of $L^{-1}$, the solution to the TPS warp. Moreover $L_k^{-1}$ is deeply connected to the bending energy of the warp. In fact, the expression ${\mathbf{\Upsilon}}^{\intercal} \mathbf{L_k^{-1}} \mathbf{\Upsilon}$ (where $\mathbf{\Upsilon} = \mathbf{\Upsilon}^0 + \mathbf{U}T)$ is exactly proportional to the bending energy of the TPS warp from the positions in $\mathbf{\Upsilon}$ to the reference landmark set😲.[^n2]


We can readily use the closed form solution for a GLS problem and get the final locations of the semilandmarks.


After moving the semilandmarks, they will most likely drift off the specimen mesh. So we may re-project them back onto the mesh to ensure that the final positions lie on the mesh.


### Modified GPA


We could modify [Generalized Procrustes Analysis](https://en.wikipedia.org/wiki/Generalized_Procrustes_analysis) or GPA routine with the above semilandmarks technique as follows.


1. Normally perform GPA on the landmark sets and get the average set.
2. Let semilandmarks slide along their tangents hyperplanes – line if curve, plane if surface – till the bending energy to the average is minimized.
3. Project the semilandmarks back onto the respective curve/surface as they likely veered off their respective structures after step 2.
4. Go back to step 1 and repeat till convergence.


We can optionally scale back the amount of sliding in each step with a dampening factor, to help near areas of high curvature.



### Optional: Why minimize bending energy?


For a rough physically motivated argument in 2D, note that the TPS map $f_{\text{tp}}$ minimizes the functional [^n3] $$V[f] = \iint \limits_{\mathbb{R}^2} J[f] dx dy = \iint \limits_{\mathbb{R}^2} (f_{xx}^2 + 2f_{xy}^2 + f_{yy}^2) dx dy$$ and the bending energy is $V[f_ {\text{tp}} ]$.


Now $J$ actually shows up in elasticity theory when studying the potential energy of an infinite thin-plate subject to perpendicular forces as follows – If $\kappa_1$ and $\kappa_2$ are the [principal curvatures](https://en.wikipedia.org/wiki/Principal_curvature) of the plate, then under certain assumptions $J$ can be physically interpreted as approximating ${\kappa_1}^2 + {\kappa_2}^2$. See [^4].


So loosely, the bending energy encodes the unavoidable "curvature energy" due to the TPS deformation. It follows that if we set our sliding step to minimize the bending energy, we’ll get the post-sliding semilandmark positions that cause the least amount of unavoidable curving.


Also [^2] describes how TPS warps are invariant under translations and rotations of both the source and target sets.





---


 Hopefully the above gives an intuitive explanation for how semilandmarks can help to minimize spurious shape differences arising from equidistant sampling.


References
----------

[^1]: Gunz P, Mitteroecker P, Bookstein FL. Semilandmarks in three dimensions. In: *Modern Morphometrics in Physical Anthropology*. Springer; 2005:73–98.
[^2]: Bookstein FL. Principal warps: Thin-plate splines and the decomposition of deformations. *IEEE Transactions on pattern analysis and machine intelligence*. 1989;11(6):567-585.
[^3]: Wang Z. LECTURE 12: VARIATIONS AND JACOBI FIELDS. <http://staff.ustc.edu.cn/~wangzuoq/Courses/16S-RiemGeom/Notes/Lec12.pdf>.
[^4]: Rohr K. *Landmark-Based Image Analysis*. Springer Netherlands; 2001. doi:[10.1007/978-94-015-9787-6](https://doi.org/10.1007/978-94-015-9787-6)
[^n1]: There is a concept of a normal Jacobi Field​​, which provides a framework to parameterize the variation "perpendicular" to a geodesic.
[^n2]: Interestingly, the bending energy matrix is a symmetric positive semidefinite matrix, much like any variance-covariance matrix. I might write more about this later.
[^n3]: over the class of maps with square integrable second derivatives
