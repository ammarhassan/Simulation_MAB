# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:44:01 2015

@author: Ammar
"""

import matplotlib.pylab as plt

def func(b):
    points = []
    js = range(1,20)
    for j in js:
        st = 2**(j-1)
        en = st * 2
        points.append(sum([b**i - b**(en) for i in range(st, en)]))
    return js, points


for b in range(9999,10000):
    b = b*1.0/10000
    tim, points = func(b)
    plt.plot(tim, points)
        