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