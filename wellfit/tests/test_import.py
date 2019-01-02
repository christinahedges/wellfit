import pytest
import warnings
import numpy as np
import wellfit as wf

def test_initialization():
    '''Ensure that all the classes can be initialized'''
    star = wf.Star()
    planet = wf.Planet(host=star)
    model = wf.Model(host=star, planets=[planet])
    model = wf.Model(host=star, planets=[planet, planet])

    time = np.arange(-1, 1, 0.0001)
    # Check the model computes
    model.compute(time=time)
