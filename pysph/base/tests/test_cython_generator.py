"""Test code for Cython code generation.
"""
import unittest
from textwrap import dedent
from math import pi, sin

from pysph.base.cython_generator import (CythonGenerator, CythonClassHelper,
    all_numeric)

def declare(*args):
    pass

class BasicEq:
    def __init__(self, hidden=None, rho=0.0, c=0.0):
        self.rho = rho
        self.c = c
        self._hidden = ['a', 'b']

class EqWithMethod(BasicEq):
    def func(self, d_idx=0, d_x=[0.0, 0.0]):
        tmp = abs(self.rho*self.c)*sin(pi*self.c)
        d_x[d_idx] = d_x[d_idx]*tmp

class EqWithReturn(BasicEq):
    def func(self, d_idx=0, d_x=[0.0, 0.0]):
        return d_x[d_idx]

class EqWithKnownTypes:
    def some_func(self, d_idx, d_p, WIJ, DWIJ):
        d_p[d_idx] = WIJ*DWIJ[0]

class EqWithMatrix:
    def func(self, d_idx, d_x=[0.0, 0.0]):
        mat = declare('matrix((2,2))')
        mat[0][0] = d_x[d_idx]
        vec = declare('matrix((3,))')
        vec[0] = d_x[d_idx]

def func_with_return(d_idx, d_x, x=0.0):
    x += 1
    return d_x[d_idx] + x

def simple_func(d_idx, d_x, x=0.0):
    d_x[d_idx] += x


class TestBase(unittest.TestCase):
    def assert_code_equal(self, result, expect):
        expect = expect.strip()
        result = result.strip()
        msg = 'EXPECTED:\n%s\nGOT:\n%s'%(expect, result)
        self.assertEqual(expect, result, msg)


class TestMiscUtils(TestBase):

    def test_all_numeric(self):
        x = [1, 2, 3.0]
        self.assertTrue(all_numeric(x))
        x = [0.0, 1, 3L]
        self.assertTrue(all_numeric(x))
        x = [0.0, 1.0, '']
        self.assertFalse(all_numeric(x))

    def test_detect_type(self):
        cases = [(('d_something', None), 'double*'),
                 (('s_something', None), 'double*'),
                 (('d_idx', 0), 'long'),
                 (('x', 1), 'long'),
                 (('s', 'asdas'), 'str'),
                 (('junk', 1.0), 'double'),
                 (('y', [0.0, 1]), 'double*'),
                 (('y', [0, 1, 0]), 'double*'),
                 (('y', None), 'object'),
                ]
        cg = CythonGenerator()
        for args, expect in cases:
            msg = 'detect_type(*%r) != %r'%(args, expect)
            self.assertEqual(cg.detect_type(*args), expect, msg)


    def test_cython_class_helper(self):
        code = ('def f(self, x):',
                '        x += 1\n        return x+1')
        c = CythonClassHelper(name='A', public_vars={'x': 'double'},
                              methods=[code])
        expect = dedent("""
        cdef class A:
            cdef public double x
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)

            def f(self, x):
                x += 1
                return x+1
        """)
        self.assert_code_equal(c.generate().strip(), expect.strip())


class TestCythonCodeGenerator(TestBase):
    def test_simple_constructor(self):
        cg = CythonGenerator()
        cg.parse(BasicEq())
        expect = dedent("""
        cdef class BasicEq:
            cdef public double c
            cdef public list _hidden
            cdef public double rho
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

    def test_simple_method(self):
        cg = CythonGenerator()
        cg.parse(EqWithMethod())
        expect = dedent("""
        cdef class EqWithMethod:
            cdef public double c
            cdef public list _hidden
            cdef public double rho
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)

            cdef inline void func(self, long d_idx, double* d_x):
                cdef double tmp
                tmp = abs(self.rho*self.c)*sin(pi*self.c)
                d_x[d_idx] = d_x[d_idx]*tmp
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

    def test_python_methods(self):
        cg = CythonGenerator(python_methods=True)
        cg.parse(EqWithMethod())
        expect = dedent("""
        cdef class EqWithMethod:
            cdef public double c
            cdef public list _hidden
            cdef public double rho
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)

            cdef inline void func(self, long d_idx, double* d_x):
                cdef double tmp
                tmp = abs(self.rho*self.c)*sin(pi*self.c)
                d_x[d_idx] = d_x[d_idx]*tmp

            cpdef py_func(self, long d_idx, double[:] d_x):
                self.func(d_idx, &d_x[0])
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

        cg.parse(EqWithReturn())
        expect = dedent("""
        cdef class EqWithReturn:
            cdef public double c
            cdef public list _hidden
            cdef public double rho
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)

            cdef inline double func(self, long d_idx, double* d_x):
                return d_x[d_idx]

            cpdef double py_func(self, long d_idx, double[:] d_x):
                return self.func(d_idx, &d_x[0])
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

        cg.parse(func_with_return)
        expect = dedent("""
        cdef inline double func_with_return(long d_idx, double* d_x, double x):
            x += 1
            return d_x[d_idx] + x

        cpdef double py_func_with_return(long d_idx, double[:] d_x, double x):
            return func_with_return(d_idx, &d_x[0], x)
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())


    def test_method_with_return(self):
        cg = CythonGenerator()
        cg.parse(EqWithReturn())
        expect = dedent("""
        cdef class EqWithReturn:
            cdef public double c
            cdef public list _hidden
            cdef public double rho
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)

            cdef inline double func(self, long d_idx, double* d_x):
                return d_x[d_idx]
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

    def test_method_with_matrix(self):
        cg = CythonGenerator()
        cg.parse(EqWithMatrix())
        expect = dedent("""
        cdef class EqWithMatrix:
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)

            cdef inline void func(self, long d_idx, double* d_x):
                cdef double mat[2][2]
                mat[0][0] = d_x[d_idx]
                cdef double vec[3]
                vec[0] = d_x[d_idx]
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

    def test_method_with_known_types(self):
        cg = CythonGenerator(known_types={'WIJ':0.0, 'DWIJ':[0.0, 0.0, 0.0]})
        cg.parse(EqWithKnownTypes())
        expect = dedent("""
        cdef class EqWithKnownTypes:
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    setattr(self, key, value)

            cdef inline void some_func(self, long d_idx, double* d_p, double WIJ, double* DWIJ):
                d_p[d_idx] = WIJ*DWIJ[0]
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

    def test_wrap_function(self):
        cg = CythonGenerator()
        cg.parse(func_with_return)
        expect = dedent("""
        cdef inline double func_with_return(long d_idx, double* d_x, double x):
            x += 1
            return d_x[d_idx] + x
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())

        cg.parse(simple_func)
        expect = dedent("""
        cdef inline void simple_func(long d_idx, double* d_x, double x):
            d_x[d_idx] += x
        """)
        self.assert_code_equal(cg.get_code().strip(), expect.strip())


if __name__ == '__main__':
    unittest.main()
