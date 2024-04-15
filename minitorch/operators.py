"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    Multiplies two floating point numbers.

    Args:
        x: A floating point number.
        y: A floating point number.

    Returns:
        The product of x and y as a floating point number.
    """
    return x * y


def id(x: float) -> float:
    """
    Identity function.

    Args:
        x: A floating point number.

    Returns:
        The input x.
    """
    return float(x)


def add(x: float, y: float) -> float:
    """
    Adds two floating point numbers.

    Args:
        x: A floating point number.
        y: A floating point number.
    Returns:
        The sum of x and y as a floating point number.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negation function.

    Args:
        x: A floating point number.

    Returns:
        Negation of x.
    """
    return -1.0 * float(x)


def lt(x: float, y: float) -> float:
    """
    Checks if x is less than y.

    Args:
        x: A floating point number.
        y: A floating point number.

    Returns:
        1.0 if x is less than y, 0.0 otherwise.
    """
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    """
    Checks if x is equal to y.

    Args:
        x: A floating point number.
        y: A floating point number.

    Returns:
        1.0 if x is less than y, 0.0 otherwise.
    """

    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    """
    Returns the maximum of x and y.

    Args:
        x: A floating point number.
        y: A floating point number.

    Returns:
        The maximum of x and y.
    """

    if x > y:
        return x
    return y


def is_close(x: float, y: float) -> float:
    """
    Checks if the absolute difference between x and y is less than 1e-2.

    Args:
        x: A floating point number.
        y: A floating point number.
    Returns:
        1.0 if the absolute difference between x and y is less than 1e-2, 0.0 otherwise.

    """
    if abs(x - y) < 1 / exp(2):
        return 1.0
    return 0.0


def sigmoid(x: float) -> float:

    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    if x > 0:
        return x
    return 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    """
    Returns the derivative of the log function at x multiplied by d.

    Args:
        x: A floating point number.
        d: A floating point number.

    Returns:
        The derivative of the log function at x multiplied by d as a floating point number.
    """
    return d * (inv(x + EPS))


def inv(x: float) -> float:
    """
    Inverse function.

    Args:
        x: A floating point number.

    Returns:
        1/x as a floating point number.
    """
    if x == 0:
        return 0.0
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """

    Multiply the derivative of the inverse function at x by d.

    Args:
        x: A floating point number.
        d: A floating point number.

    Returns:
        d*(-1/x^2) as a floating point number.

    """
    return d * -1.0 * (inv(x**2))


def relu_back(x: float, d: float) -> float:
    if x > 0:
        return d
    return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def new_function(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return new_function


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Takes a list of floats and returns a new list with each element negated.

    Args:
        ls: A list of floating point numbers.

    Returns:
        [ -ls[0], -ls[1], ..., -ls[n]]

    """
    neg_map = map(neg)
    return neg_map(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def new_func(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return new_func


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Takes two lists of floats and returns a new list with each element added.

    Args:
        ls1: A list of floating point numbers.
        ls2: A list of floating point numbers.

    Returns:
        [ ls1[0] + ls2[0], ls1[1] + ls2[1], ..., ls1[n] + ls2[n]]

    """
    add_zip = zipWith(add)
    return add_zip(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def new_func(ls: Iterable[float]) -> float:
        last_value = start
        for x in ls:
            last_value = fn(last_value, x)
        return last_value

    return new_func


def sum(ls: Iterable[float]) -> float:
    """
    Takes a list of floats and returns the sum of all elements using reduce and add.

    Args:
        ls: A list of floating point numbers.

    Returns:
        The sum of all elements in the list as a floating point number.
    """
    sum_reduce = reduce(add, 0)
    return sum_reduce(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Takes a list of floats and returns the product of all elements using reduce and mul.

    Args:
        ls: A list of floating point numbers.
    Returns:
        The product of all elements in the list as a floating point number.
    """
    mul_reduce = reduce(mul, 1)
    return mul_reduce(ls)
