import numpy as np
def select_non_zero_constituents(event):
    # Select only constituents with non-zero pT
    return event[event[:, 0] != 0]


def parse_event(event):
    # Returns MET, electron and muon constituents with non-zero pT

    assert event.ndim == 2
    non_zero_part = select_non_zero_constituents(event)
    # Make sure the input has a dimension of 2

    met_part = non_zero_part[non_zero_part[:, 3] == 1]
    electron_part = non_zero_part[non_zero_part[:, 3] == 2]
    muon_part = non_zero_part[non_zero_part[:, 3] == 3]
    jet_part = non_zero_part[non_zero_part[:, 3] == 4]

    return met_part, electron_part, muon_part, jet_part

def embed_dict(embed_points, type):
    output_dict = {"type": [type] * len(embed_points)}
    for i in range(embed_points.shape[1]):
        output_dict["Dimension {}".format(i)] = embed_points[:, i]
    return output_dict

def sample_pairs(n_events, n_pairs):
    # np.random.choice can be extremely slow, use randint instead
    pairs = np.random.randint(0, n_events, (n_pairs, 2))
    # remove pairs with same index since we don't want to compare the same event
    bad_samples_idx = np.where(pairs[:, 0] == pairs[:, 1])
    for i in bad_samples_idx:
        pairs[i] = np.random.choice(n_events, 2, replace=False)
    return pairs

def sample_matrix(n_events, pairs):
    matrix = np.zeros((n_events, n_events))
    for pair in pairs:
        matrix[pair[0], pair[1]] += 1
    return matrix