import datetime
from os import stat

import numpy as np
import pytest

from ..array import StateVector, Matrix, SPDCovarianceMatrix
from ..state import GaussianState, State
from ..extendedstate import Extent, EllipticalExtent, GaussianInverseWishartExtendedState, InverseWishartExtent, ExtendedState, EllipticalExtendedState


def test_extent():
    extent_variable = Matrix([[1, 2], [3, 4]])
    ext = Extent(extent_variable)
    assert np.array_equal(ext.extent_variable, extent_variable)


def test_ellipticalextent():
    with pytest.raises(ValueError):
        extent_variable = Matrix([[1, 2], [2, 3]])
        EllipticalExtent(extent_variable)
    
    with pytest.raises(ValueError):
        extent_variable = Matrix([[0, 2], [-2, 0]])
        EllipticalExtent(extent_variable)


def test_inversewishartextent():
    scale_matrix = Matrix([[2, 2], [2, 6]])
    degrees_of_freedom = 5
    invwish_ext = InverseWishartExtent(scale_matrix, degrees_of_freedom)
    assert np.array_equal(invwish_ext.mean, Matrix([[1, 1], [1, 3]]))


def test_extendedstate():
    with pytest.raises(TypeError):
        ExtendedState()

    # Test extended state initiation without timestamp
    state = State(StateVector([[0], [1]]))
    extent = Extent(Matrix([[1, 2], [3, 4]]))
    ext_state = ExtendedState(state, extent)
    assert np.array_equal(ext_state.state.state_vector, state.state_vector)
    assert np.array_equal(ext_state.extent.extent_variable, extent.extent_variable)

    # Test extended state initiation with timestamp
    timestamp = datetime.datetime.now()
    ext_state = ExtendedState(state, extent, timestamp=timestamp)
    assert ext_state.timestamp == timestamp


def test_ellipticalextendedstate():
    with pytest.raises(TypeError):
        EllipticalExtendedState()
    
    # Test positive definetness 
    state_vector = State(StateVector([[0], [1]]))
    extent_matrix = Extent(Matrix([[1, 2], [2, 4]]))
    with pytest.raises(ValueError):
        EllipticalExtendedState(state_vector, extent_matrix)

    # Test symmetry
    extent_matrix = Extent(Matrix([[1, 1], [0, 4]]))
    with pytest.raises(ValueError):
        EllipticalExtendedState(state_vector, extent_matrix)


def test_gaussianinversewishartextendedstate():

    # Test extent mean
    state_vector = StateVector([[0], [1]])
    covar = SPDCovarianceMatrix([[1, 0], [0, 4]])
    scale_matrix = SPDCovarianceMatrix([[4, 2], [2, 6]])
    degrees_of_freedom = 5
    state = GaussianState(state_vector, covar)
    extent = InverseWishartExtent(scale_matrix, degrees_of_freedom)
    giw_state = GaussianInverseWishartExtendedState(state, extent)
    assert np.array_equal(giw_state.mean[1], np.array([[2, 1], [1, 3]]))
