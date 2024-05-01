import numpy as np
from scipy.spatial.distance import cdist
from ot.lp import emd_c, check_result


# Ref: https://github.com/SangeonPark/ToyJetGenerator/blob/main/optimal_transport/emd.py

def process_event_np(event):
    event = np.array(event)
    # Sort constituents in event by pT in descending order
    # event = event[np.argsort(event[:, 0])[::-1]]
    pts = event[:, 0]
    coords = event[:, 1:3]
    return np.ascontiguousarray(pts), np.ascontiguousarray(coords)  # Return as contiguous arrays for C compatibility


def check_shape(event):
    assert event.ndim == 2 and event.shape[-1] == 4, "Event shape must be (n, 4)"


def emd_pot(source_event, target_event, norm=False, R=1.0, return_flow=False, n_iter_max=100000):
    # Compute energy  mover's distance between two events using python ot library 

    check_shape(source_event)
    check_shape(target_event)

    source_pTs, source_coords = process_event_np(source_event)
    target_pTs, target_coords = process_event_np(target_event)
    source_total_pT, target_total_pT = source_pTs.sum(), target_pTs.sum()
    source_coords = source_coords
    target_coords = target_coords

    # if norm, then we normalize the pts to 1
    if norm:
        source_pTs /= source_total_pT
        target_pTs /= target_total_pT
        thetas = cdist(source_coords, target_coords) / R
        rescale = 1.0

    # implement the EMD in Eq. 1 of the paper by adding an appropriate extra particle
    else:
        pTdiff = target_total_pT - source_total_pT
        if pTdiff > 0:
            source_pTs = np.hstack((source_pTs, pTdiff))
            source_coords_extra = np.vstack((source_coords, np.zeros(source_coords.shape[1], dtype=np.float64)))
            thetas = cdist(source_coords_extra, target_coords) / R
            thetas[-1, :] = 1.0

        elif pTdiff < 0:
            target_pTs = np.hstack((target_pTs, -pTdiff))
            target_coords_extra = np.vstack((target_coords, np.zeros(target_coords.shape[1], dtype=np.float64)))
            thetas = cdist(source_coords, target_coords_extra) / R
            thetas[:, -1] = 1.0

        # in this case, the pts were exactly equal already so no need to add a particle
        else:
            thetas = cdist(source_coords, target_coords) / R

        # change units for numerical stability
        rescale = max(source_total_pT, target_total_pT)

    # # Psuedorapidity-azimuth distance matrix
    # dist_matrix = cdist(source_coords, target_coords, 'euclidean')

    flow_matrix, cost, _, _, result_code = emd_c(source_pTs / rescale, target_pTs / rescale, thetas, n_iter_max, True)
    check_result(result_code)

    if return_flow:
        return cost * rescale, flow_matrix * rescale
    else:
        return cost * rescale
