Mathematical Groups
===================

We will talk a lot about groups and Lie groups in the theory sections, so it is important to have some
background on what a group is and why they are interesting. The first step is understanding up the
notion of a *set*. Without getting into the details (and there are many details), a set is a collection of objects
which may or may not have a relationship to one another. The integers are a archetypal example of a *set* which we
write as:

.. math::
    Z = {..., -2, -1, 0, 1, 2, 3, ...}

With these special { brackets to denote that what we are looking at is a *set* and not a vector or other object. If I
wish to start adding some of these numbers together I would find some, perhaps obvious, but nonetheless profound
properties.

1. **Associativity**
    .. math::
        (a + b) + c = a + (b + c)

    for any a, b, c in Z. So for any element in the set of integers, this condition will be true.

2. **Identity Element**
    .. math::
        a + 0 = a

    for all a in Z. 0 in this case is the element upon which adding to any others, leaves it unchanged.

3. **Inverse Element**
    .. math::
        a + -a = 0

    This is a little more complicated in principle, but for addition is quite intuitive. There will exist some inverse
    element (the negative element for addition) such that when I apply an operation (in this case addition) these
    elements they will result in identity element (0 in this case).

We are finally ready to discuss what a group is! Phew. When we take the set of integers Z and say that the addition
*operator* obeys the above criteria on the elements of Z, then we can say that the combination of Z and + is a
group. We would write this as (Z, +) to denote that we are talking about the set of integers and the addition operator.

In order to generalize the notion of a group, we need to re-write these *axioms* to be applicable to more than just
addition. The standard notion is to write an operation as :math:`\cdot`.

1. **Associativity**
    .. math::
        (a \cdot b) \cdot c = a \cdot (b \cdot c)

2. **Identity Element**
    .. math::
        a \cdot i = a

3. **Inverse Element**
    .. math::
        a \cdot b = i

The final point in the general definition ofa group is that the application of the operator :math:`\cdot` on the
group elements will result in an element which is also in the group. We might write this for some group
(G, :math:`\hat{A}`):

    .. math::
        \hat{A} : a, b \in G \rightarrow c \in G

or more simply,

    .. math::
       \hat{A} : G \rightarrow G

This axiom is called **closure** or **totality** and is fundamental to the definition of a group.

Symmetry Groups
---------------
In the example above we discussed a group consisting of integers and an operation, namely, addition. However, the
elements of the set do not need to be numbers or in fact, scalar objects, they can themselves be operations. This can
include functions and matrices. Take rotating points on a square by :math:`90` for example. This rotation, or
any rotation of integer multiples of :math:`90` will lead to the square looking the same. If we take the
rotations of vertices on a square as elements of a set denoting them as :math:`\hat{R}_{\theta}`, and define the
operator :math:`\circ` such that :math:`\hat{R}_{90}\circ\hat{R}_{90}` denoted a rotation of
square vertices be :math:`180` (i.e. :math:`90` twice), we could say that this set of rotations and the
operation :math:`\circ` denoted a group, perhaps written (R, :math:`\circ`). To extend the definition, we would also
say that this was a symmetry group, as the application of these operations on the vertices of a square will result in
an effectively unchanged object. I will note that there are some intricacies I have ignored here, in particular that
the :math:`\theta` values should be confined to a set of the sort {0, 90, 180, 270}, but I hope that the concept is
clear. There is a gif in progress to describe this better!

Abelian (Commutative) Groups
----------------------------
While there won't be too much reference to Abelian groups I will mention them here to be complete. Commutative operators
are something we deal with in all mathematics but only come across after a certain level of mathematical education. To
illustrate the idea we will stick some examples. Let's define two matrices A, and B such that:

.. math::

    A = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \\ \\
    B = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{pmatrix}

Now take a vector :math:`\mathbf{r} = [1, 1, 1]` and let's apply these matrices in different orders and see what
comes out.

.. math::
    A \circ B \mathbf{x} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} \\ \\
    B \circ A \mathbf{x} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}

Essentially this boils down to :math:`A\circ B \neq B\circ A`. In mathematical jargon we would say that A does not
commute with B. This is quite common, particularly with matrices as in general, matrix multiplication is not
commutative, something that most of us will have learned at some point in high school (albeit likely with different
language). So why are we discussing this property? Well, if you manage to collect a set of operations
(matrices for example), along with this :math:`\circ` relation, and all of these operators commute with one another
while also fulfilling the conditions listed above, you have an Abelian group! In the case of rotations, it is clear that
for three dimensions it will not be possibe to have an Abelian group as if you rotate through one plane and then another,
you will end up in a different place than if you had performed the operations in reverse. (Again, a gif is on the way).

Lie Groups
----------
Bluntly put, a Lie group is a group that also happens to be a differentiable :ref:`manifold <manifold>`. If you haven't
had time to read through the full documentation page of :ref:`manifold <manifold>` then I will quickly summarize. A
manifold broadly is a topology (surface for example) that locally resembles Euclidean space (the flat x, y, z axis we
are all familiar with). If the manifold in question is smooth, continuous, and differentiable, and we have a group
describing operations like addition and multiplication (along with their inverses), then we find ourselves with a Lie
group. A nice property of Lie groups is that contain a Lie algebra. This is however not a topic for the Groups section
of the theory and if interested, you should go and see the :ref:`algebra` part of the documentation. When we are
studying operations in physics, we will often come across certain symmetries. These can arise in all areas from
classical mechanics and quantum mechanics through general relativity and string theory. These symmetries allow us to
make some assumptions about what it is we are studying. If the operators we are studying appear to form a Lie group,
we can then use all the information and properties of Lie groups as mathematical tools to study our system. This is,
it a vastly simplified summary, the benefit of identifying and understanding Lie groups. On a mathematical level, the
benefits of Lie groups arise mostly in their underlying algebra, and so I will leave it to the algebra section to
outline these.

References
**********
