import copy
import datetime
import typing
from collections import abc
from typing import MutableSequence, Optional, Any
from numbers import Real

from ..base import Property
from .base import Type
from .array import Matrix, SPDCovarianceMatrix
from .state import State, GaussianState


class Extent(Type):
    """Extent type.

    An extent and a timestamp."""
    timestamp: datetime.datetime = Property(
        default=None, doc="Timestamp of the extent. Default None.")
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
    def from_extent(extent: 'Extent', *args: Any, target_type: Optional[typing.Type] = None,
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

class CreatableFromExtent:
    class_mapping = {}

    def __init_subclass__(cls, **kwargs):
        bases = cls.__bases__
        if CreatableFromExtent in bases:
            # Direct subclasses should not be added to the class mapping, only subclasses of
            # subclasses
            return
        if len(bases) != 2:
            raise TypeError('A CreatableFromExtent subclass must have exactly two superclasses')
        base_class, extent_type = cls.__bases__
        if not issubclass(base_class, CreatableFromExtent):
            raise TypeError('The first superclass of a CreatableFromExtent subclass must be a '
                            'CreatableFromExtent (or a subclass)')
        if not issubclass(extent_type, Extent):
            # Non-extent subclasses do not need adding to the class mapping, as they should not
            # be created from Extents
            return
        if base_class not in CreatableFromExtent.class_mapping:
            CreatableFromExtent.class_mapping[base_class] = {}
        CreatableFromExtent.class_mapping[base_class][extent_type] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_extent(
            cls,
            extent: Extent,
            *args: Any,
            target_type: Optional[type] = None,
            **kwargs: Any) -> 'Extent':
        # Handle being initialised with extent sequence
        if isinstance(extent, ExtentMutableSequence):
            extent = extent.extent
        try:
            extent_type = next(type_ for type_ in type(extent).mro()
                              if type_ in CreatableFromExtent.class_mapping[cls])
        except StopIteration:
            raise TypeError(f'{cls.__name__} type not defined for {type(extent).__name__}')
        if target_type is None:
            target_type = CreatableFromExtent.class_mapping[cls][extent_type]

        return target_type.from_extent(extent, *args, **kwargs, target_type=target_type)


class ExtentMutableSequence(Type, abc.MutableSequence):

    extents: MutableSequence[Extent] = Property(
        default=None,
        doc="The initial list of extents. Default `None` which initialises with empty list.")

    def __init__(self, extents=None, *args, **kwargs):
        if extents is None:
            extents = []
        elif not isinstance(extents, abc.Sequence):
            # Ensure extents is a list
            extents = [extents]
        super().__init__(extents, *args, **kwargs)

    def __len__(self):
        return self.extents.__len__()

    def __setitem__(self, index, value):
        return self.extents.__setitem__(index, value)

    def __delitem__(self, index):
        return self.extents.__delitem__(index)

    def __getitem__(self, index):
        if isinstance(index, slice) and (
                isinstance(index.start, datetime.datetime)
                or isinstance(index.stop, datetime.datetime)):
            items = []
            for extent in self.extents:
                try:
                    if index.start and extent.timestamp < index.start:
                        continue
                    if index.stop and extent.timestamp >= index.stop:
                        continue
                except TypeError as exc:
                    raise TypeError(
                        'both indices must be `datetime.datetime` objects for'
                        'time slice') from exc
                items.append(extent)
            return ExtentMutableSequence(items[::index.step])
        elif isinstance(index, datetime.datetime):
            for extent in reversed(self.extents):
                if extent.timestamp == index:
                    return extent
            else:
                raise IndexError('timestamp not found in extents')
        elif isinstance(index, slice):
            return ExtentMutableSequence(self.extents.__getitem__(index))
        else:
            return self.extents.__getitem__(index)

    def __getattribute__(self, name):
        # This method is called if we try to access an attribute of self. First we try to get the
        # attribute directly, but if that fails, we want to try getting the same attribute from
        # self.extent instead. If that, in turn,  fails we want to return the error message that
        # would have originally been raised, rather than an error message that the Extent has no
        # such attribute.
        #
        # An alternative mechanism using __getattr__ seems simpler (as it skips the first few lines
        # of code, but __getattr__ has no mechanism to capture the originally raised error.
        try:
            # This tries first to get the attribute from self.
            return Type.__getattribute__(self, name)
        except AttributeError as original_error:
            if name.startswith("_"):
                # Don't proxy special/private attributes to `extent`, just raise the original error
                raise original_error
            else:
                # For non _ attributes, try to get the attribute from self.extent instead of self.
                try:
                    my_extent = Type.__getattribute__(self, 'extent')
                    return getattr(my_extent, name)
                except AttributeError:
                    # If we get the error about 'Extent' not having the attribute, then we want to
                    # raise the original error instead
                    raise original_error

    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        property_name = self.__class__.extents._property_name
        inst.__dict__[property_name] = copy.copy(self.__dict__[property_name])
        return inst

    def insert(self, index, value):
        return self.extents.insert(index, value)

    @property
    def extent(self):
        return self.extents[-1]

    def last_timestamp_generator(self):
        """Generator yielding the last extent for each timestamp

        This provides a method of iterating over a sequence of extents,
        such that when multiple extents for the same timestamp exist,
        only the last extent is yielded. This is particularly useful in
        cases where you may have multiple :class:`~.Update` extents for
        a single timestamp e.g. multi-sensor tracking example.

        Yields
        ------
        Extent
            A extent for each timestamp present in the sequence.
        """
        extent_iter = iter(self)
        current_extent = next(extent_iter)
        for next_extent in extent_iter:
            if next_extent.timestamp > current_extent.timestamp:
                yield current_extent
            current_extent = next_extent
        yield current_extent


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
    
    Represents Inverse-Wishart extent."""
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
        "A tuple representing number of dimensions in the extended state."
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