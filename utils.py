def select_non_zero_constituents(event):
    # Select only constituents with non-zero pT
    return event[event[:, 0] != 0]


def parse_event(event):
    # Returns MET, electron and muon constituents with non-zero pT

    assert event.ndim == 2 #Input should have a dimension of 2
    non_zero_part = select_non_zero_constituents(event)
    # Make sure the input has a dimension of 2
    

    met_part = non_zero_part[non_zero_part[:, 3] == 1]
    electron_part = non_zero_part[non_zero_part[:, 3] == 2]
    muon_part = non_zero_part[non_zero_part[:, 3] == 3]
    jet_part = non_zero_part[non_zero_part[:, 3] == 4]

    return met_part, electron_part, muon_part, jet_part