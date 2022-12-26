#!/usr/bin/env python3
# File       : test_func.py
# Description: Test cases for Func class
# Copyright 2022 Harvard University. All Rights Reserved


import pytest
import numpy as np
import autodiff as ad
from autodiff.func import Func

class TestFunc:
    '''Test class for Func constructor and methods'''

    def test_init(self):
        '''Test for the Func construtor'''

        f = lambda x: x+1
        num_in = 1
        num_out = 1
        with pytest.raises(AssertionError):
            Func(f, 3.2, 1)
            Func(f, 3, 1.8)
            Func(f, 0, 5)
            Func(f, 8, 0)
        
        my_function = Func(f, num_in, num_out)
        assert my_function.func == f
        assert my_function.num_inputs == num_in
        assert my_function.num_outputs == num_out
    
    def test_repr(self):
        '''Test for repr of Func class instances'''

        f = lambda x: x+1
        num_in = 1
        num_out = 1
        my_function = Func(f, num_in, num_out)
        assert repr(my_function) == f"Function {f} with {num_in} input(s) and {num_out} output(s)"

    def test_call(self):
        '''Test for call of Func class instances'''

        f = lambda x: x+1
        my_function = Func(f, 1, 1)
        assert my_function(1) == 2
        def f2(x, y):
            return x+1, y**2
        my_function = Func(f2, 2, 2)
        assert np.array_equal(my_function([1, 4]), np.array([2, 16]))
        with pytest.raises(ValueError):
            my_function([4])

    def test_jacobian(self):
        '''Test for jacobian generating method of Func class instances'''

        f = lambda x: x+1
        def ff(x):
            return x+1, x+4
        num_in = 1
        num_out = 1
        my_function = Func(f, num_in, num_out)
        my_function2 = Func(f, num_in, num_out + 1)
        my_function3 = Func(ff, num_in, num_out)

        with pytest.raises(AssertionError):
            my_function.jacobian(point = 'bana')
            my_function2.jacobian(point = [6])
            my_function3.jacobian(point = [6])
        with pytest.raises(ValueError):
            my_function.jacobian(point = [6,5])

        assert my_function.jacobian(6) == 1
        assert my_function.jacobian(6, reverse=True) == 1

        def fff(x,y):
            return y * x**2, 5 * x + ad.sin(y)
        my_function4 = Func(fff, 2, 2)
        assert np.array_equal(my_function4.jacobian([2.5,np.pi]), np.array([[2*2.5*np.pi, 2.5**2],[5, np.cos(np.pi)]]))
        assert np.array_equal(my_function4.jacobian([2.5,np.pi], reverse=True), np.array([[2*2.5*np.pi, 2.5**2],[5, np.cos(np.pi)]]))
        v4, j4 = my_function4.jacobian([2.5,np.pi], feval=True) 
        assert np.array_equal(v4, np.array([6.25*np.pi, 12.5]))
        assert np.array_equal(j4, np.array([[2*2.5*np.pi, 2.5**2],[5, np.cos(np.pi)]]))

        def f4(x,y):
            return ad.tan(y), ad.cos(x)
        my_function4 = Func(f4, 2, 2)
        assert np.allclose(my_function4.jacobian([np.pi,np.pi]), np.array([[0, 1],[0, 0]]))
        assert np.allclose(my_function4.jacobian([np.pi,np.pi], reverse=True), np.array([[0, 1],[0, 0]]))

        def f5(x,y):
            return ad.exp(x)+1, ad.log(y)
        my_function5 = Func(f5, 2, 2)
        assert np.allclose(my_function5.jacobian([2,4]), np.array([[np.exp(2), 0],[0, 1/4]]))
        assert np.allclose(my_function5.jacobian([2,4], reverse=True), np.array([[np.exp(2), 0],[0, 1/4]]))
        with pytest.raises(ValueError):
            my_function5.jacobian([4, -1])
            my_function5.jacobian([4, -1], reverse=True)

        def f6(x,y):
            return ad.log_base(x, 10)+1, ad.arcsin(y)-6
        my_function6 = Func(f6, 2, 2)
        assert np.allclose(my_function6.jacobian([2,0]), np.array([[1/(2*np.log(10)), 0],[0, 1]]))
        assert np.allclose(my_function6.jacobian([2,0], reverse=True), np.array([[1/(2*np.log(10)), 0],[0, 1]]))

        def f7(x,y):
            return ad.log_base(10, 10)+ad.arccos(x), ad.arctan(y)-6
        my_function7 = Func(f7, 2, 2)
        v7, j7 = my_function7.jacobian([.5,1], feval=True)
        assert np.array_equal(v7, np.array([1+np.arccos(.5), np.arctan(1)-6])) 
        assert np.allclose(j7, np.array([[-1/(np.sqrt(.75)), 0],[0, .5]]))
        v7, j7 = my_function7.jacobian([.5,1], feval=True, reverse=True)
        assert np.array_equal(v7, np.array([1+np.arccos(.5), np.arctan(1)-6])) 
        assert np.allclose(j7, np.array([[-1/(np.sqrt(.75)), 0],[0, .5]]))

        def f8(x):
            return ad.sinh(x)-5, ad.cosh(x)*ad.tanh(x) 
        my_function8 = Func(f8, 1, 2)
        v8, j8 = my_function8.jacobian([.5], feval=True)
        assert np.array_equal(v8, np.array([np.sinh(.5)-5, np.cosh(.5)*np.tanh(.5)])) 
        assert np.allclose(j8, np.array([[np.cosh(.5)], [np.cosh(.5)]]))
        v8, j8 = my_function8.jacobian([.5], feval=True, reverse=True)
        assert np.array_equal(v8, np.array([np.sinh(.5)-5, np.cosh(.5)*np.tanh(.5)])) 
        assert np.allclose(j8, np.array([[np.cosh(.5)], [np.cosh(.5)]]))

        def f9(x):
            return ad.logistic(x), ad.sqrt(x)*5*ad.sin(x**2)
        my_function9 = Func(f9, 1, 2)
        v9, j9 = my_function9.jacobian([.5], feval=True)
        assert np.array_equal(v9, np.array([1/(1+np.exp(-.5)), 5*np.sqrt(.5)*np.sin(.25)])) 
        assert np.allclose(j9, np.array([[(1/(1+np.exp(-.5)))*(1-1/(1+np.exp(-.5)))], [5*(np.sin(.25)+np.cos(.25))/(2*np.sqrt(.5))]]))
        v9, j9 = my_function9.jacobian([.5], feval=True, reverse=True)
        assert np.array_equal(v9, np.array([1/(1+np.exp(-.5)), 5*np.sqrt(.5)*np.sin(.25)])) 
        assert np.allclose(j9, np.array([[ad.logistic(.5)*(1-ad.logistic(.5))], [5*(np.sin(.25)+np.cos(.25))/(2*np.sqrt(.5))]]))

        def f10(x):
            return 2**-x
        my_function10 = Func(f10, 1, 1)
        v10, j10 = my_function10.jacobian([.5], feval=True)
        assert v10 == pow(2, -.5)
        assert j10 == -2**-.5 * np.log(2)
        v10, j10 = my_function10.jacobian([.5], feval=True, reverse=True)
        assert v10 == pow(2, -.5)
        assert j10 == -2**-.5 * np.log(2)

    def test_trace(self):
        '''Test for function and derivative evaluation method of Func class instances'''

        f = lambda x: x+1
        def ff(x):
            return 3*x + 1, x**2 + 4
        num_in = 1
        num_out = 1
        my_function = Func(f, num_in, num_out)
        my_function2 = Func(f, num_in, num_out+1)
        my_function3 = Func(ff, num_in, num_out)
        
        with pytest.raises(AssertionError):
            my_function.trace(point = 'bana', seed_vector = [4])
            my_function.trace(point = np.array([4]), seed_vector = 9)
            my_function2.trace([2.5],[1])
            my_function3.trace([2.5],[1])
        with pytest.raises(ValueError):
            my_function.trace(point = [6], seed_vector = [4,2])
        with pytest.raises(ValueError):
            my_function.trace(point = [6,2], seed_vector = [4])
            my_function.trace(point = [6,3], seed_vector = [4,2])
        
        fff = lambda x: ad.exp(x**2) + 1
        my_function4 = Func(fff, num_in, num_out)
        out_val, out_deriv = my_function4.trace(3,[1])
        assert np.array_equal(out_val,np.array([fff(3)]))
        assert np.array_equal(out_deriv,np.array([2*3*np.exp(3**2) * 1]))
        
        def ffff(x,y,z):
            return z * y * x**2 + ad.cos(x), 5 * x + y - z 
        
        my_function5 = Func(ffff, 3, 2)
        out_val, out_deriv = my_function5.trace([3,2,4],[1,0,0])
        assert np.array_equal(out_val, np.array(ffff(3,2,4)))
        assert np.array_equal(out_deriv, np.array([4*2*2*3 - np.sin(3), 5]))

    def test_graph(self):
        '''Test to ensure proper display of computational graph'''

        def fofx(x,y):
            return  x + 2*y
        f = Func(fofx,2,1)
        p = [5,4.2]
        graph = f.graph(p)

        with pytest.raises(ValueError):
            gp = f.graph([5,4,2])
        
        def fx(x):
            return x**3 + 4
        ff = Func(fx,1,1)
        graph = ff.graph(3)
        