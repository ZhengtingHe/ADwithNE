import matplotlib.pyplot as plt
import numpy as np
from utils import parse_event
import seaborn as sns

from emd import separate_particles

marker_scale = 30


def create_eta_phi_plane():
    # Create a 2D eta-phi plane
    plt.figure(figsize=(5, 5))
    plt.xlim(-5, 5)
    plt.ylim(-3.5, 3.5)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\phi$')
    plt.grid(True)


def create_legend():
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
    create_legend()
    plt.show()


def plot_optimal_transport(source_event, target_event, flow_matrix, emd, verbose=False):
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
    for i, (receiving_coord, sending_coord) in enumerate(zip(receiving_coords, sending_coords)):
        flow = flow_matrix[index_send[i], index_receive[i]]

        plt.arrow(sending_coord[0], sending_coord[1],  # arrow base
                  receiving_coord[0] - sending_coord[0], receiving_coord[1] - sending_coord[1],
                  head_width=0.2, head_length=0.2, shape='left',
                  fc='#FFD700', ec='#FFD700')  # arrow span

        if verbose:
            print(f'From: {np.round(sending_coord, 2)} to {np.round(receiving_coord, 2)} with flow {flow:.2f}')

    create_legend()
    plt.title(f'EMD: {emd:.2f}, Total flow: {len(index_send)}')
    plt.show()

def plot_particle(event_coord, event_pt, type, state='source'):
    particle_colors = np.array([
        ["#DA70D6", "#800080"],
        ["#FFA07A", "#FF4500"],
        ["#D3D3D3", "#708090"]
    ])
    event_coord = event_coord[event_pt != 0]
    event_pt = event_pt[event_pt != 0]

    color_state = 1 if state == 'target' else 0
    plot_colors = particle_colors[:, color_state]

    markers = ['^', 's', 'D']
    labels = ['Electrons', 'Muons', 'Jets']
    plt.scatter(
        event_coord[:, 0], event_coord[:, 1],
        color=plot_colors[type],
        marker=markers[type],
        s=event_pt * marker_scale,
        alpha=0.5,
        label=state + labels[type]
    )

    for eta, phi, pt in zip(event_coord[:, 0], event_coord[:, 1], event_pt):
        plt.text(eta, phi, "{:.2f}".format(pt))

def plot_separate_optimal_transport(source_event, target_event, e_matrix, mu_matrix, jet_matrix, total_emd, verbose=False):
    MET_source_pt, _, electron_source_pts, electron_source_coords, muon_source_pts, muon_source_coords, jet_source_pts, jet_source_coords = separate_particles(source_event)
    MET_target_pt, _, electron_target_pts, electron_target_coords, muon_target_pts, muon_target_coords, jet_target_pts, jet_target_coords = separate_particles(target_event)
    
    separate_source_pt = [electron_source_pts, muon_source_pts, jet_source_pts]
    separate_target_pt = [electron_target_pts, muon_target_pts, jet_target_pts]

    separate_target_coords = [electron_source_coords, muon_source_coords, jet_source_coords]
    separate_source_coords = [electron_target_coords, muon_target_coords, jet_target_coords]

    separate_matrix = [e_matrix, mu_matrix, jet_matrix]
    plt.figure(figsize=(15, 5))
    for i in range(3):
        if separate_matrix[i] is not None:

            source_coords = separate_source_coords[i]
            target_coords = separate_target_coords[i]

            source_pts = separate_source_pt[i]
            target_pts = separate_target_pt[i]
            flow_matrix = separate_matrix[i]

            plt.subplot(1, 3, i + 1)

            plot_particle(source_coords, source_pts, i, state='source')
            plot_particle(target_coords, target_pts, i, state='target')
            plt.xlim(-5, 5)
            plt.ylim(-3.5, 3.5)
            plt.xlabel(r'$\eta$')
            plt.ylabel(r'$\phi$')
            plt.grid(True)
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
            for i, (receiving_coord, sending_coord) in enumerate(zip(receiving_coords, sending_coords)):
                flow = flow_matrix[index_send[i], index_receive[i]]

                plt.arrow(sending_coord[0], sending_coord[1],  # arrow base
                        receiving_coord[0] - sending_coord[0], receiving_coord[1] - sending_coord[1],
                        head_width=0.2, head_length=0.2, shape='left',
                        fc='#FFD700', ec='#FFD700')  # arrow span

                if verbose:
                    print(f'From: {np.round(sending_coord, 2)} to {np.round(receiving_coord, 2)} with flow {flow:.2f}')

            create_legend()
    plt.suptitle("Total EMD: {:.2f}".format(total_emd))
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


def plot_train_history(dfhistory, title="Training History", yscale=False):
    x = np.arange(len(dfhistory["train_loss"]))
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, dfhistory["train_MAPE"], '#1f77b4', label="train MAPE")
    ax1.plot(x, dfhistory["val_MAPE"], '#aec7e8', label="val MAPE")
    # ax1.plot(x, dfhistory["train_Embed Ratio"], '#196619', label="train Embed Ratio")
    # ax1.plot(x, dfhistory["val_Embed Ratio"], '#98df8a', label="val Embed Ratio")
    ax1.legend(loc="upper left")
    ax2.plot(x, dfhistory["train_loss"], '#d62728', label="tarin loss")
    ax2.plot(x, dfhistory["val_loss"], '#f7b6d2', label="val loss")
    ax2.legend(loc="upper right")
    if yscale:
        ax1.set_yscale(yscale)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAPE of distence')
    ax2.set_ylabel('loss')

    ax1.grid()
    plt.title(title)
    plt.show()


def downsample_and_visualize_pairplot(df, sample_size, palette=None):
    # Downsample the DataFrame
    df_sampled = df.sample(sample_size, random_state=114514)

    # Create pair plot using Seaborn
    sns.pairplot(df_sampled, hue="type", kind="kde", palette=palette)

    # Show the plot using Matplotlib
    plt.show()

def plot_roc_curve(fpr_dict, tpr_dict, auc_dict, title):
    for key in fpr_dict.keys():
        fpr = fpr_dict[key]
        tpr = tpr_dict[key]
        auc_roc = auc_dict[key]
        plt.plot(fpr, tpr, label='Lambda = {} (AUC = {:.2f})'.format(key, auc_roc))
    # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(auc_roc), lw=2)    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR') 
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_lambda(lambda_mean_dict, lambda_std_dict):
    x = np.array([float(key) for key in lambda_mean_dict.keys()])
    y = np.array([lambda_mean_dict[key] for key in lambda_mean_dict.keys()])
    yerr = np.array([lambda_std_dict[key] for key in lambda_std_dict.keys()])
    plt.errorbar(x, y, yerr=yerr, fmt='o', color='black', ecolor='lightgray', elinewidth=3)
    true_x = np.linspace(0, np.max(x), 10)
    plt.plot(true_x, true_x, '--')
    plt.xlabel("True Lambda")
    plt.ylabel("Estimated Lambda")
    plt.title("Estimated Lambda vs True Lambda")
    plt.show()
