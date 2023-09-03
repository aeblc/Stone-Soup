import numpy as np
import pytest

from ..groundtruth import ExtendedStateGroundTruthPath, ExtentGroundTruthPath, GroundTruthExtent, GroundTruthState, GroundTruthPath, CategoricalGroundTruthState, \
    CompositeGroundTruthState


def test_groundtruthpath():
    empty_path = GroundTruthPath()

    assert len(empty_path) == 0

    groundtruth_path = GroundTruthPath([
        GroundTruthState(np.array([[0]])) for _ in range(10)])

    assert len(groundtruth_path) == 10

    state1 = GroundTruthState(np.array([[1]]))
    groundtruth_path.append(state1)
    assert groundtruth_path[-1] is state1
    assert groundtruth_path.states[-1] is state1

    state2 = GroundTruthState(np.array([[2]]))
    groundtruth_path[0] = state2
    assert groundtruth_path[0] is state2

    groundtruth_path.remove(state1)
    assert state1 not in groundtruth_path

    del groundtruth_path[0]
    assert state2 not in groundtruth_path

def test_extentgroundtruthpath():
    empty_path = ExtentGroundTruthPath()

    assert len(empty_path) == 0

    groundtruth_path = ExtentGroundTruthPath([
        GroundTruthExtent(np.array([[1, 0], [0, 1]])) for _ in range(10)])

    assert len(groundtruth_path) == 10

    state1 = GroundTruthExtent(np.array([[2, 0], [0, 9]]))
    groundtruth_path.append(state1)
    assert groundtruth_path[-1] is state1
    assert groundtruth_path.extents[-1] is state1

    state2 = GroundTruthExtent(np.array([[9, 0], [0, 1]]))
    groundtruth_path[0] = state2
    assert groundtruth_path[0] is state2

    groundtruth_path.remove(state1)
    assert state1 not in groundtruth_path

    del groundtruth_path[0]
    assert state2 not in groundtruth_path

def test_extendedstategroundtruthpath():
    empty_path = ExtendedStateGroundTruthPath()

    assert len(empty_path.state_ground_truth) == 0
    assert len(empty_path.extent_ground_truth) == 0

    with pytest.raises(ValueError):
        groundtruth_path = ExtendedStateGroundTruthPath(
            states = [GroundTruthState(np.array([[1], [1]])) for _ in range(8)],
            extents = [GroundTruthExtent(np.array([[1, 0], [0, 1]])) for _ in range(10)])

def test_composite_groundtruth():
    sub_state1 = GroundTruthState([0], metadata={'colour': 'red'})
    sub_state2 = GroundTruthState([1], metadata={'speed': 'fast'})
    sub_state3 = CategoricalGroundTruthState([0.6, 0.4], metadata={'shape': 'square'})
    state = CompositeGroundTruthState(sub_states=[sub_state1, sub_state2, sub_state3])
    assert state.metadata == {'colour': 'red', 'speed': 'fast', 'shape': 'square'}
