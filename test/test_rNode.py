#!/usr/bin/env python3
# File       : test_rNode.py
# Description: Test cases for rNode class
# Copyright 2022 Harvard University. All Rights Reserved
import pytest
import numpy as np
from autodiff.rNode import rNode

class TestrNode:
    """Test class for rNode objects"""


    def test_init(self):
        '''Test for the rNode construtor'''
        
        node = rNode(2)
        nodef = rNode(3.42)
        
        assert isinstance(node,rNode)
        assert node.val == 2
        assert node.der == 0
        assert node.oper == ''

        assert isinstance(nodef,rNode)
        assert nodef.val == 3.42
        assert nodef.der == 0
        assert nodef.oper == ''


    def test_repr(self):
        '''Test for repr of rNode class instances'''
        
        rNode.increment == 0
        node = rNode(2)
        a = repr(node)

    def test_neg(self):
        '''Test for unary negation of rNode class instances'''
        
        node = rNode(2)
        neg = -node
        assert neg.val == -2

    def test_add(self):
        '''Test for addition with rNode class instances'''
        
        node1 = rNode(2)
        node2 = rNode(4)
        node3 = node1 + node2
        assert node3.val == 6

        node4 = node1 + 1
        assert node4.val == 3

        node5 = node1 + 2.5
        assert node5.val == 4.5
        assert node3.oper == '+'
        assert node3.backward_func is not None

        
        with pytest.raises(TypeError):
            node1 + '1'
            node1 + True
    
    def test_radd(self):
        '''Test for reflective addition with rNode class instances'''
        
        node1 = rNode(2)
        node2 =   1 + node1
        node3 = node1 + 1

        assert node2.val == node3.val
        assert node2.oper == node3.oper
        
        with pytest.raises(TypeError):
            '1' + node1
            False + node1


    def test_sub(self):
        '''Test for subtraction with rNode class instances'''

        node1 = rNode(2)
        node2 = rNode(4)
        node3 = node1 - node2
        assert node3.val == -2

        node4 = node1 - 1
        assert node4.val == 1


        node5 = node1 - 2.5
        assert node5.val == -0.5

        assert node3.backward_func is not None

        
        with pytest.raises(TypeError):
            node1 - '1'
            node1 - True


    def test_rsub(self):
        """Test for reflective subtraction with rNode class instances"""

        node1 = rNode(2)

        node4 = 1 - node1
        assert node4.val == -1


        node5 = 2.5 - node1
        assert node5.val == 0.5

        assert node4.backward_func is not None

        with pytest.raises(TypeError):
            '1' - node1 
            True -  node1

    def test_mul(self):
        '''Test for multiplication with rNode class instances'''
        
        node1 = rNode(2)
        node2 = rNode(4)
        node3 = node1 * node2
        assert node3.val == 8

        node4 = node1 * 2
        assert node4.val == 4

        node5 = node1 * 2.5
        assert node5.val == 5
        assert node3.oper == 'x'
        assert node3.backward_func is not None

        
        with pytest.raises(TypeError):
            node1 * '1'
            node1 * True
        
    def test_rmul(self):
        '''Test for reflective multiplication with rNode class instances'''
        
        node1 = rNode(2)
        node2 =   2 * node1
        node3 = node1  * 2

        assert node2.val == node3.val
        assert node2.oper == node3.oper
        
        with pytest.raises(TypeError):
            '2' * node1
            False * node1

    def test_pow(self):
        """Test for power with rNode class instances"""

        node1 = rNode(2)
        node2 = rNode(4)
        node3 = node1**node2
        assert node3.val == 16

        node4 = node1**2
        assert node4.val == 4


        assert node3.oper == '^'
        assert node3.backward_func is not None
        
        with pytest.raises(TypeError):
            node1**'1'
            node1** True

    def test_rpow(self):
        """Test for exponentiation with rNode clas instances"""
        node1 = rNode(2)
        node3 = 2**node1

        assert isinstance(node3,rNode)
        assert node3.val == 4
        
        assert node3.backward_func is not None

        with pytest.raises(TypeError):
            '2' ** node1
            False ** node1
    
    def test_truediv(self):

        node1 = rNode(2)
        node2 = rNode(4)
        node3 = node1/ node2
        assert node3.val == 0.5

        node4 = node1 /2
        assert node4.val == 1

        node5 = node1 / 2.5
        assert node5.val == 2/2.5
        assert node3.oper == '/'
        assert node3.backward_func is not None
        
        with pytest.raises(TypeError):
            node1/'1'
            node1/True

    def test_rtruediv(self):

        node1 = rNode(2)


        node4 = 2/node1
        assert node4.val == 1

        node5 =  2.5/node1
        assert node5.val == 2.5/2
        assert node5.backward_func is not None
        
        with pytest.raises(TypeError):
            '1'/node1
            True/node1

        
    def test_sort_nodes(self):

        x1 = rNode(2)
        x2 = rNode(3)

        f = lambda x,y : x + y - 2

        out = f(x1,x2)
        outout = out._sort_nodes()

        assert len(outout) == 5
        for node in outout:
            assert isinstance(node, rNode)
    
    def test_graph_object(self):

        x1 = rNode(2)
        x2 = rNode(3)

        f = lambda x,y : x + y - 2

        out = f(x1,x2)

        outout = out.graph_object()

        assert len(outout) == 2
        assert isinstance(outout,tuple)
        assert isinstance(outout[0],set)
        assert isinstance(outout[1],set)
        assert len(outout[0]) == 5
        assert len(outout[1]) == 4
        for tup in outout[1]:
            assert isinstance(tup, tuple)

    def test_backward(self):
        rNode.increment= 0
        x1 = rNode(2)
        x2 = rNode(3)

        f = lambda x,y : 2*x + y - 2

        out = f(x1,x2)
        outout = out.backward(2)
        assert outout[0] == 2.0
        assert outout[1] == 1.0



    