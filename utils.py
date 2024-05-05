import toml
import os
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


def embed_dict(embed_points, event_type):
    output_dict = {"type": [event_type] * len(embed_points)}
    for i in range(embed_points.shape[1]):
        output_dict["Dimension {}".format(i)] = embed_points[:, i]
    return output_dict


def load_toml_config(key):
    module_path = os.path.dirname(__file__)
    with open(os.path.join(module_path, 'config.toml'), 'r') as f:
        config = toml.load(f)
    return config[key]