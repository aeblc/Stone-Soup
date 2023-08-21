import datetime
import typing
from typing import Optional, Any
from numbers import Real

from ..base import Property
from .base import Type
from .array import Matrix, SPDCovarianceMatrix
from .state import State, GaussianState


class Extent(Type):
    """Extent type.

    An extent and a timestamp."""
    timestamp: datetime.datetime = Property(
        default=None, doc="Timestamp of the state. Default None.")
    extent_variable: Matrix = Property(doc="Extent variable.")

    def __init__(self, extent_variable, *args, **kwargs):
        # Don't cast away subtype of extent_variable if not necessary
        if extent_variable is not None \
                and not isinstance(extent_variable, Matrix):
            extent_variable = Matrix(extent_variable)
        super().__init__(extent_variable, *args, **kwargs)

    @property
    def ndim(self):
        """The number of dimensions represented by the extent_variable."""
        return self.extent_variable.shape[0]

    @staticmethod
    def from_state(extent: 'Extent', *args: Any, target_type: Optional[typing.Type] = None,
                   **kwargs: Any) -> 'Extent':
        if target_type is None:
            target_type = type(extent)

        args_property_names = {
            name for n, name in enumerate(target_type.properties) if n < len(args)}

        new_kwargs = {
            name: getattr(extent, name)
            for name in type(extent).properties.keys() & target_type.properties.keys()
            if name not in args_property_names and name not in kwargs}

        new_kwargs.update(kwargs)

        return target_type(*args, **new_kwargs)


class EllipticalExtent(Extent):
    """EllipticalExtent type.

    An extent with symmetric positive definite matrix as extent variable."""
    def __init__(self, extent_variable, *args, **kwargs):
        # Don't cast away subtype of extent_matrix if not necessary
        if extent_variable is not None \
                and not isinstance(extent_variable, SPDCovarianceMatrix):
            extent_variable = SPDCovarianceMatrix(extent_variable)
        super().__init__(extent_variable, *args, **kwargs)


class InverseWishartExtent(EllipticalExtent):
    """InverseWishartExtent type.
    
    Represents Inverse-Wishart state."""
    scale_matrix: SPDCovarianceMatrix = Property(doc="Scale matrix of the Inverse-Wishart distributed random matrix.")
    degrees_of_freedom: Real = Property(doc="Degrees of freedom of the Inverse-Wishart distributed random matrix.")

    def __init__(self, scale_matrix, degrees_of_freedom, *args, **kwargs):
        if degrees_of_freedom <= scale_matrix.shape[0] + 1:
            raise ValueError("Degrees of freedom must be greated than scale matrix dimension.")
        super().__init__(scale_matrix, degrees_of_freedom, *args, **kwargs)
    
    @property
    def mean(self):
        """Mean of the Inverse-Wishart distribution"""
        return self.scale_matrix/(self.degrees_of_freedom - self.scale_matrix.shape[0] - 1)

    @property
    def extent_variable(self):
        """Just returns the distribution mean as the extent variable."""
        return self.mean


class ExtendedState(Type):
    """ExtendedState type.

    Most general extended state type, which only has time, state vector and an extent
    variable."""

    timestamp: datetime.datetime = Property(
        default=None, doc="Timestamp of the state. Default None.")
    state: State = Property(doc="State component of the extended state.")
    extent: Extent = Property(doc="Extent component of the extended state.")

    def __init__(self, state, extent, *args, **kwargs):
        if not isinstance(state, State):
            state = State(state)
        if not isinstance(extent, Extent):
            extent = Extent(extent)
        super().__init__(state, extent, *args, **kwargs)

    @property
    def ndim(self):
        "A tuple representing number of dimensions in the extrended state."
        return (self.state.ndim, self.extent.ndim)


class EllipticalExtendedState(ExtendedState):
    """EllipticalExtendedState type.

    An extended state with symmetric positive definite matrix as extent variable."""

    extent: EllipticalExtent = Property(doc="Symmetric positive definite covariance matrix as the elliptical extent variable.")

    def __init__(self, state, extent, *args, **kwargs):
        if not isinstance(extent, EllipticalExtent):
            extent = EllipticalExtent(extent)   
        super().__init__(state, extent, *args, **kwargs)


class GaussianInverseWishartExtendedState(EllipticalExtendedState):
    """GaussianInverseWishartState type.

    Represents Gaussian Inverse-Wishart state."""

    state: GaussianState = Property(doc="Gaussian state representation with a mean vector and a covariance matrix.")
    extent: InverseWishartExtent = Property(doc="Inverse-Wishart extent matrix with a scale matrix and a degrees of freedom parameter.")

    def __init__(self, state, extent, *args, **kwargs):
        if not isinstance(state, GaussianState):
            state = GaussianState(state.state_vector, state.covar)
        if not isinstance(extent, InverseWishartExtent):
            extent = InverseWishartExtent(extent.scale_matrix, extent.degrees_of_freedom)
        super().__init__(state, extent, *args, **kwargs)

    @property
    def mean(self):
        return (self.state.mean, self.extent.mean)