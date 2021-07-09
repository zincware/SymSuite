Algebra
=======
Most people who have found their way to this web-page have likely come across
the notion of an Algebra in their lifetime. This is because it is introduced to
us at a very early stage in the context of solving systems of equations of
arbitrary complexity, e.g.

.. math::

   2\cdot x + 1 = 5

please solve for x.

While this is in fact Algebra the concept goes far beyond replacing some numbers
with letters and torturing students for hours on end with systems of linear
equations.

*More to come*

Lie Algebra
^^^^^^^^^^^
A type of algebra we will come across in this package quite a lot is known as
Lie algebra. A Lie algebra is vector space :math:`\mathcal{g}` together with an
operation referred to as the Lie bracket. This operation should be an alternating
bi-linear map (:math:`\mathcal{g} \circ \mathcal{g} \rightarrow \mathcal{g}`)
satisfying the Jacobi identity:

.. math::

   [X, [Y, X]] + [Y, [Z, X]] + [Z, [X, Y]] = 0.

What this identity implies is that we can commute elements of the Lie algebra
[A, B] and generate either 0 or a new element C. C can take on two possibilities:

1.) C is a linear combination of A and B
2.) C is a new element of the algebra.

While there is the possibility that this commutation property can go on
indefinitely, typically a finite number of these elements are found. This is a
defining property of a Lie algebra.

