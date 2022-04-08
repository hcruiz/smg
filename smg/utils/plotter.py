import os
import numpy as np
import matplotlib.pyplot as plt

size = 7
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('axes', labelsize=size)
plt.rc('legend', fontsize=size)
plt.rc("savefig", dpi=300, format='png')


def output_hist(outputs, bins=100, save_dir=None, show=False):
    plt.figure()
    plt.suptitle('Output Histogram')
    plt.hist(outputs, bins)
    plt.ylabel('Counts')
    plt.xlabel('Outputs (nA)')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'output_distribution'))
    if show:
        plt.show()
    else:
        plt.close()


def plot_errors(targets, prediction, error, save_dir=None, name='test_error', show=False):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(targets, prediction, '.')
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    targets_and_prediction_array = np.concatenate((targets, prediction))
    min_out = np.min(targets_and_prediction_array)
    max_out = np.max(targets_and_prediction_array)
    plt.plot(np.linspace(min_out, max_out), np.linspace(min_out, max_out), 'k')
    plt.title(f'Predicted vs True values:\n MSE {np.mean(error ** 2,axis=0)}')
    plt.subplot(1, 2, 2)
    plt.hist(np.reshape(error, error.size), 500)
    x_lim = 0.25 * np.max([np.abs(error.min()), error.max()])
    plt.xlim([-x_lim, x_lim])
    plt.title('Scaled error histogram')
    if save_dir:
        fig_loc = os.path.join(save_dir, name)
        plt.savefig(fig_loc, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_error_vs_output(targets, error, save_dir=None, name='test_error_vs_output', show=False):
    plt.figure()
    plt.plot(targets, error, '.')
    plt.plot(np.linspace(targets.min(), targets.max(),
             len(error)), np.zeros_like(error))
    plt.title('Error vs Output')
    plt.xlabel('Output')
    plt.ylabel('Error')
    if save_dir:
        fig_loc = os.path.join(save_dir, name)
        plt.savefig(fig_loc, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_all(targets, predictions, save_dir=None, show=True):
    assert targets.shape == predictions.shape, \
        f"Shape mismatch: targets {targets.shape}, predictions {predictions.shape}"
    errors = predictions - targets
    plot_error_vs_output(targets, errors, save_dir=save_dir, show=show)
    plot_errors(targets, predictions, errors,
                save_dir=save_dir, show=show)
