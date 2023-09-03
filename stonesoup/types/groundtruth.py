import uuid
from typing import MutableSequence, MutableMapping, Sequence

from stonesoup.types.base import Type


from .state import State, StateMutableSequence, CategoricalState, CompositeState
from .extendedstate import ExtendedState, Extent, ExtentMutableSequence
from ..base import Property


class GroundTruthState(State):
    """Ground Truth State type"""
    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class GroundTruthExtent(Extent):
    """Ground Truth Extent type"""
    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, extent_variable, *args, **kwargs):
        super().__init__(extent_variable, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class GroundTruthExtendedState(ExtendedState):
    """ Ground Truth Extended State Type """
    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, state, extent, *args, **kwargs):
        super().__init__(state, extent, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}

class CategoricalGroundTruthState(GroundTruthState, CategoricalState):
    """Categorical Ground Truth State type"""


class GroundTruthPath(StateMutableSequence):
    """Ground Truth Path type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states: MutableSequence[GroundTruthState] = Property(
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")
    id: str = Property(
        default=None,
        doc="The unique path ID. Default `None` where random UUID is "
            "generated.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())


class ExtentGroundTruthPath(ExtentMutableSequence):
    """Extent Ground Truth Path type

    A :class:`~.ExtentMutableSequence` representing an extent track.
    """

    extents: MutableSequence[GroundTruthExtent] = Property(
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")
    id: str = Property(
        default=None,
        doc="The unique path ID. Default `None` where random UUID is "
            "generated.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())

class ExtendedStateGroundTruthPath():
    """ Extended State Ground Truth Path type """ 
    
    def __init__(self, states = None, extents = None, *args, **kwargs):
        if states is not None:
            self.state_ground_truth = GroundTruthPath(states, *args, **kwargs)
        else:
            self.state_ground_truth = GroundTruthPath(*args, **kwargs)
        
        if extents is not None:
            self.extent_ground_truth = ExtentGroundTruthPath(extents, *args, **kwargs)
        else:
            self.extent_ground_truth = ExtentGroundTruthPath(*args, **kwargs)
            
        self.id = str(uuid.uuid4())
        try:
            assert(len(self.state_ground_truth) == len(self.extent_ground_truth))
        except:
            raise(ValueError("State and extent ground truth must have the same length."))
        

class CompositeGroundTruthState(CompositeState):
    """Composite ground truth state type.

    A composition of ordered sub-states (:class:`GroundTruthState`) existing at the same timestamp,
    representing a true object with a state for (potentially) multiple, distinct state spaces.
    """

    sub_states: Sequence[GroundTruthState] = Property(
        doc="Sequence of sub-states comprising the composite state. All sub-states must have "
            "matching timestamp and `metadata` attributes. Must not be empty.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def metadata(self):
        """Combined metadata of all sub-detections."""
        metadata = dict()
        for sub_state in self.sub_states:
            metadata.update(sub_state.metadata)
        return metadata


GroundTruthState.register(CompositeGroundTruthState)  # noqa: E305
