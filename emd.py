import numpy as np
from scipy.spatial.distance import cdist
from ot.lp import emd_c, check_result


# Ref: https://github.com/SangeonPark/ToyJetGenerator/blob/main/optimal_transport/emd.py

def process_event_np(event, particle_type_scale=0, particle_one_hot=True):
    event = np.array(event)
    # Sort constituents in event by pT in descending order
    # event = event[np.argsort(event[:, 0])[::-1]]
    pts = event[:, 0]
    coords = event[:, 1:]
    if particle_one_hot:
        type_encoding = np.vstack((np.zeros([1,4]), np.eye(4)))
        coords = np.hstack((coords[:, :2], type_encoding[coords[:,2].astype(int)]))
    coords[:, 2:] *= particle_type_scale
    return np.ascontiguousarray(pts), np.ascontiguousarray(coords)  # Return as contiguous arrays for C compatibility

def separate_particles(event):
    MET_pt = event[0, 0]
    MET_coords = event[0, 1:3]

    electron_pts = event[1:5, 0]
    electron_coords = event[1:5, 1:3]

    muon_pts = event[5:9, 0]
    muon_coords = event[5:9, 1:3]

    jet_pts = event[9:, 0]
    jet_coords = event[9:, 1:3]
    return MET_pt, MET_coords, electron_pts, electron_coords, muon_pts, muon_coords, jet_pts, jet_coords


def check_shape(event):
    assert event.ndim == 2 and event.shape[-1] == 4, "Event shape must be (n, 4)"


def emd_pot(source_event, target_event, norm=False, return_flow=False, n_iter_max=100000, particle_type_scale=0, particle_one_hot=True):
    # Compute energy  mover's distance between two events using python ot library 

    check_shape(source_event)
    check_shape(target_event)

    source_pTs, source_coords = process_event_np(source_event, particle_type_scale, particle_one_hot=particle_one_hot)
    target_pTs, target_coords = process_event_np(target_event, particle_type_scale, particle_one_hot=particle_one_hot)
    source_total_pT, target_total_pT = source_pTs.sum(), target_pTs.sum()
    source_coords = source_coords
    target_coords = target_coords
    # R ≥1 2θmax, where θmax is the maximum attainable angular distance between particles
    R = np.sqrt(4 ** 2 + np.pi ** 2 + (2 * particle_type_scale) ** 2)

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
    
def ot_within_type(source_pt, target_pt, source_coords, target_coords, n_iter_max=100000):
    source_sum_pt = source_pt.sum()
    target_sum_pt = target_pt.sum()
    R = np.sqrt(4 ** 2 + np.pi ** 2)
    pt_diff = target_sum_pt - source_sum_pt
    if pt_diff > 0:
        source_pt = np.hstack((source_pt, pt_diff))
        source_coords_extra = np.vstack((source_coords, np.zeros(source_coords.shape[1], dtype=np.float64)))
        thetas = cdist(source_coords_extra, target_coords) / R
        thetas[-1, :] = 1.0
    elif pt_diff < 0:
        target_pt = np.hstack((target_pt, -pt_diff))
        target_coords_extra = np.vstack((target_coords, np.zeros(target_coords.shape[1], dtype=np.float64)))
        thetas = cdist(source_coords, target_coords_extra) / R
        thetas[:, -1] = 1.0
    else:
        thetas = cdist(source_coords, target_coords) / R
    rescale = max(source_sum_pt, target_sum_pt)
    _, cost, _, _, result_code = emd_c(source_pt / rescale, target_pt / rescale, thetas, n_iter_max, True)
    check_result(result_code)
    return cost * rescale
    

def sep_emd(source_event, target_event, n_iter_max=100000):
    MET_source_pt, MET_source_coords, electron_source_pts, electron_source_coords, muon_source_pts, muon_source_coords, jet_source_pts, jet_source_coords = separate_particles(source_event)
    MET_target_pt, MET_target_coords, electron_target_pts, electron_target_coords, muon_target_pts, muon_target_coords, jet_target_pts, jet_target_coords = separate_particles(target_event)

    R = np.sqrt(4 ** 2 + np.pi ** 2)

    MET_EMD = abs(MET_source_pt - MET_target_pt)


    # Make sure that there are at least 1 specific particle in the source or target
    if electron_source_pts[0] != 0 or electron_target_pts[0] != 0:
        electron_EMD = ot_within_type(electron_source_pts, electron_target_pts, electron_source_coords, electron_target_coords, n_iter_max)

    else:
        electron_EMD = 0
    
    
    if muon_source_pts[0] != 0 or muon_target_pts[0] != 0:
        muon_EMD = ot_within_type(muon_source_pts, muon_target_pts, muon_source_coords, muon_target_coords, n_iter_max)

    else:
        muon_EMD = 0
    
    
    if jet_source_pts[0] != 0 or jet_target_pts[0] != 0:
        jet_EMD = ot_within_type(jet_source_pts, jet_target_pts, jet_source_coords, jet_target_coords, n_iter_max)

    else:
        jet_EMD = 0
    
    total_EMD = MET_EMD + electron_EMD + muon_EMD + jet_EMD
    return total_EMD



