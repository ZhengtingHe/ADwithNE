import matplotlib.pyplot as plt
import numpy as np

from utils import parse_event

marker_scale = 30


def create_eta_phi_plane():
    # Create a 2D eta-phi plane
    plt.figure(figsize=(5, 5))
    plt.xlim(-5, 5)
    plt.ylim(-3.5, 3.5)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\phi$')
    plt.grid(True)


def create_legent():
    legend = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
    for handle in legend.legend_handles:
        handle.set_sizes([30.0])


def plot_constituents(event, state='source'):
    particle_colors = np.array([
        ["#89CFF0", "#007BFF"],
        ["#DA70D6", "#800080"],
        ["#FFA07A", "#FF4500"],
        ["#D3D3D3", "#708090"]
    ])

    color_state = 1 if state == 'target' else 0
    plot_colors = particle_colors[:, color_state]

    met_part, electron_part, muon_part, jet_part = parse_event(event)

    particle_parts = [met_part, electron_part, muon_part, jet_part]
    markers = ['o', '^', 's', 'D']
    labels = ['MET', 'Electrons', 'Muons', 'Jets']

    for i, part in enumerate(particle_parts):
        if part.size:
            plt.scatter(
                part[:, 1], part[:, 2],
                color=plot_colors[i],
                marker=markers[i],
                s=part[:, 0] * marker_scale,
                alpha=0.5,
                label=state + labels[i]
            )

            for eta, phi, pt in zip(part[:, 1], part[:, 2], part[:, 0]):
                plt.text(eta, phi, "{:.2f}".format(pt))




def plot_event_cloud(event):
    # Plot single event in the eta-phi plane
    create_eta_phi_plane()
    plot_constituents(event, state='')
    create_legent()
    plt.show()


def plot_optimal_transport(source_event, target_event, flow_matrix, emd):
    create_eta_phi_plane()
    # Plot source and target distributions
    plot_constituents(source_event, state='source')
    plot_constituents(target_event, state='target')
    # Get coordinates and pTs of source and target events
    source_pts = source_event[:, 0]
    target_pts = target_event[:, 0]
    source_coords = source_event[:, 1:3]
    target_coords = target_event[:, 1:3]
    # Create extra particles if needed
    source_total_pt = source_pts.sum()
    target_total_pt = target_pts.sum()
    pt_diff = target_total_pt - source_total_pt
    if pt_diff > 0:
        source_coords = np.vstack((source_coords, np.zeros(source_coords.shape[1], dtype=np.float64)))
        plt.scatter(0, 0, color="#CCFF00", marker='P', s=pt_diff * marker_scale, alpha=0.5, label='extra_source')
    elif pt_diff < 0:
        target_coords = np.vstack((target_coords, np.zeros(target_coords.shape[1], dtype=np.float64)))
        plt.scatter(0, 0, color="#006400", marker='P', s=-pt_diff * marker_scale, alpha=0.5, label='extra_target')
    # Take only non-zero flow values
    index_send, index_receive = np.nonzero(flow_matrix)
    # Plot arrows between source and target particles
    sending_coords = source_coords[index_send]
    receiving_coords = target_coords[index_receive]
    for receiving_coord, sending_coord in zip(receiving_coords, sending_coords):
        print("From: ", sending_coord, "To: ", receiving_coord)
        plt.arrow(sending_coord[0], sending_coord[1],  # arrow base
                  receiving_coord[0] - sending_coord[0], receiving_coord[1] - sending_coord[1],
                  head_width=0.1, fc='#FFD700', ec='#FFD700')  # arrow span

    create_legent()
    plt.title(f'EMD: {emd:.2f}')
    plt.show()


def plot_hists(events, title):
    # Plot histograms of pT, eta and phi for MET, electrons, muons and jets
    events = np.array(events).reshape(-1, 4)
    met_part, electron_part, muon_part, jet_part = parse_event(events)
    particles = [met_part, electron_part, muon_part, jet_part]
    variables = ['pT', r'$\eta$', r'$\phi$']
    particle_names = ['MET', 'Electrons', 'Muons', 'Jets']
    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F']

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))

    for i in range(4):
        for j in range(3):
            axes[i, j].hist(particles[i][:, j], bins=50, color=colors[j])
            if j == 0:
                axes[i, j].set_yscale('log')  # Log scale for pT
            axes[i, j].grid(True)

    for ax, col in zip(axes[0], variables):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], particle_names):
        ax.set_ylabel(row, rotation=0)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
