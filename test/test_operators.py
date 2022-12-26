# File       : test_operators.py
# Description: Test cases elementary operators
# Copyright 2022 Harvard University. All Rights Reserved.
import pytest
import numpy as np
import autodiff as ad
from autodiff.dualnumber import DualNumber

def test_sin():
    '''Test sin operator for various input types'''
    z1 = DualNumber(1,2)
    z2 = ad.sin(z1)
    assert z2.real == np.sin(1)
    assert z2.dual == 2*np.cos(1)
    assert ad.sin(4) == np.sin(4)
    assert ad.sin(0.5) == np.sin(0.5)
    with pytest.raises(TypeError):
        ad.sin("1")

def test_cos():
    '''Test cos operator for various input types'''
    z1 = DualNumber(1,2)
    z2 = ad.cos(z1)
    assert z2.real == np.cos(1)
    assert z2.dual == -2*np.sin(1)
    assert ad.cos(4) == np.cos(4)
    assert ad.cos(0.5) == np.cos(0.5)
    with pytest.raises(TypeError):
        ad.cos("1")

def test_tan():
    '''Test tan operator for various input types'''
    z1 = DualNumber(1,2)
    z2 = ad.tan(z1)
    assert z2.real == np.tan(1)
    assert z2.dual == 2/(np.cos(1)**2)
    assert ad.tan(4) == np.tan(4)
    assert ad.tan(0.5) == np.tan(0.5)
    with pytest.raises(TypeError):
        ad.tan("1")

def test_exp():
    '''Test exp operator for various input types'''
    z1 = DualNumber(1,2)
    z2 = ad.exp(z1)
    assert z2.real == np.exp(1)
    assert z2.dual == 2*np.exp(1)
    assert ad.exp(4) == np.exp(4)
    assert ad.exp(0.5) == np.exp(0.5)
    with pytest.raises(TypeError):
        ad.exp("1")

def test_log():
    '''Test log operator for various input types'''
    z1 = DualNumber(1,2.)
    z2 = ad.log(z1)
    assert z2.real == np.log(1)
    assert z2.dual == 2.
    assert ad.log(4) == np.log(4)
    assert ad.log(0.5) == np.log(0.5)
    with pytest.raises(TypeError):
        ad.log("1")
    with pytest.raises(ValueError):
        ad.log(-5)

def test_log_base():
    '''Test log_base operator for various input types'''
    assert ad.log_base(5, 7) == np.log(5)/np.log(7)
    with pytest.raises(TypeError):
        ad.log_base("1", 8)
        ad.log_base(4, "1")

def test_arcsin():
    '''Test arcsin operator for various input types'''
    assert ad.arcsin(.2) == np.arcsin(.2)
    with pytest.raises(TypeError):
        ad.arcsin("1")
    with pytest.raises(ValueError):
        ad.arcsin(5)

def test_arccos():
    '''Test arccos operator for various input types'''
    assert ad.arccos(.2) == np.arccos(.2)
    with pytest.raises(TypeError):
        ad.arccos("1")
    with pytest.raises(ValueError):
        ad.arccos(5)

def test_arctan():
    '''Test arctan operator for various input types'''
    assert ad.arctan(.2) == np.arctan(.2)
    with pytest.raises(TypeError):
        ad.arctan("1")

def test_sinh():
    '''Test sinh operator for various input types'''
    assert ad.sinh(.2) == np.sinh(.2)
    with pytest.raises(TypeError):
        ad.sinh("1")

def test_cosh():
    '''Test cosh operator for various input types'''
    assert ad.cosh(.2) == np.cosh(.2)
    with pytest.raises(TypeError):
        ad.cosh("1")

def test_tanh():
    '''Test tanh operator for various input types'''
    assert ad.tanh(.2) == np.tanh(.2)
    with pytest.raises(TypeError):
        ad.tanh("1")

def test_logistic():
    '''Test logistic operator for various input types'''
    assert ad.logistic(5) == 1/(1+np.exp(-5))
    with pytest.raises(TypeError):
        ad.cosh("1")

def test_sqrt():
    '''Test sqrt operator for various input types'''
    assert ad.sqrt(.2) == np.sqrt(.2)
    with pytest.raises(TypeError):
        ad.sqrt("1")