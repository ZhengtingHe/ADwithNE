import matplotlib.pyplot as plt
import numpy as np


from utils import parse_event


def plot_event_cloud(event):
    met_part, electron_part, muon_part, jet_part = parse_event(event)

    plt.figure(figsize=(5, 5))
    plt.xlim(-5, 5)
    plt.ylim(-3.5, 3.5)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\phi$')

    plt.scatter(met_part[:, 1], met_part[:, 2], color='red', marker='^', s=met_part[:, 0], label='MET')

    if electron_part.size:
        plt.scatter(electron_part[:, 1], electron_part[:, 2], color='blue', marker='o', s=electron_part[:, 0],
                    label='Electrons')

    if muon_part.size:
        plt.scatter(muon_part[:, 1], muon_part[:, 2], color='green', marker='s', s=muon_part[:, 0], label='Muons')

    if jet_part.size:
        plt.scatter(jet_part[:, 1], jet_part[:, 2], color='purple', marker='*', s=jet_part[:, 0], label='Jets')

    plt.legend()
    plt.grid(True)
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
                axes[i, j].set_yscale('log') #Log scale for pT
            axes[i, j].grid(True)

    for ax, col in zip(axes[0], variables):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], particle_names):
        ax.set_ylabel(row, rotation=0)

    
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
