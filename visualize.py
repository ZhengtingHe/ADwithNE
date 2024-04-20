import matplotlib.pyplot as plt

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
