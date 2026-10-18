import numpy as np

from bench.public_history_diverse import select_quartet


def test_diversity_selection_avoids_duplicates_and_does_not_take_prediction_scores():
    labels = np.zeros((2,5,16),np.int8)
    labels[:,2] = 1
    labels[:,3] = 2
    labels[:,4,8:] = 1
    selected,distance,score = select_quartet(labels)
    assert selected == (0,2,3,4)
    assert distance[0,1] == 0
    assert score[0] == .5
    assert np.allclose(distance,distance.T)
