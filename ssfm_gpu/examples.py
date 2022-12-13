import numpy as np
import matplotlib.pyplot as plt
# import ssfm_gpu
import tensorflow as tf
from .propagation import tf_fiber_propogate
from scipy.fft import fft, ifft, fftfreq, fftshift
import functools
from datetime import datetime


def execution_time(func):
    @functools.wraps(func)
    def wrapper_execution_time(*args, **kwargs):
        start_time = datetime.now()
        value = func(*args, **kwargs)
        end_time = datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
        print("Function [" + func.__name__ + "] execution took", execution_time, "ms")
        return value
    return wrapper_execution_time


def get_energy(signal, t_span):
    return np.mean(np.power(np.absolute(signal), 2)) * t_span


def get_gauss_pulse(amplitude, t, tau, z=0., beta2=0.):
    z_ld = z / tau ** 2 * abs(beta2)
    a_z = amplitude / np.sqrt(1 - 1.0j * z_ld * np.sign(beta2))

    return a_z * np.exp(-0.5 / (1 + z_ld ** 2) * np.power(t / tau, 2) * (1.0 + 1.0j * z_ld))


def get_pulse_nonlinear(signal, gamma, z):
    return signal * np.exp(1.0j * gamma * np.power(np.abs(signal), 2) * z)


def check_energy(signal, t_span, spectrum):
    energy_signal = np.mean(np.power(np.absolute(signal), 2))
    energy_spectrum = np.mean(np.power(np.absolute(spectrum), 2)) / len(signal)
    if abs(energy_signal - energy_spectrum) > 1e-14:
        print("Error, energy is different: ", abs(energy_signal - energy_spectrum))

    return energy_signal, energy_spectrum


@execution_time
def example_gauss_pulse(nt=2 ** 12, t_span=32., nz=2 ** 10, z_prop=1.0, beta2=-1.0, gamma=1.0):

    # nt = 2 ** 12
    # t_span = 32.
    dt = t_span / nt
    t = np.array([(i - nt / 2) * dt for i in range(nt)])
    # print(t)

    # signal = 2.7 / np.cosh(np.multiply(t, 10))

    # beta2 = -1.0
    # gamma = 1.0
    signal = get_gauss_pulse(10.0, t, 1.0, z=0, beta2=beta2)
    # signal = get_soliton_pulse(t, 1.0, 2, beta2, gamma)
    energy_init = get_energy(signal, t_span)

    # z_prop = 1.0
    # signal_prop = fiber_propogate_high_order(signal, t_span, z_prop, 2 ** 8, 0, 1.0)
    signal_prop = tf_fiber_propogate(tf.cast(signal, tf.complex128), t_span, z_prop, nz, gamma=gamma, beta2=beta2, alpha=0, beta3=0)
    signal_prop = signal_prop.numpy()
    signal_end = get_gauss_pulse(10.0, t, 1.0, z=z_prop, beta2=beta2)
    # signal_end = get_pulse_nonlinear(signal, gamma, z_prop)

    energy_end = get_energy(signal_prop, t_span)

    # band = 2 * np.pi
    # n = len(signal)
    # dw = band / t_span
    w = np.array([(i - nt / 2) * (2. * np.pi / t_span) for i in range(nt)])

    # --- Plot input pulse shape and spectrum
    spect = np.power(np.abs(fftshift(fft(signal))), 2)  # input spectrum of Fourier transform
    # spect = spect / np.max(spect)  # normalize
    # freq = fftshift(w) / (2 * np.pi)  # freq. array
    freq = w / (2 * np.pi)  # freq. array

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, np.power(np.absolute(signal), 2), 'blue')
    axs[0].plot(t, np.power(np.absolute(signal_prop), 2), 'red')
    axs[0].set_xlim(-t_span / 2, t_span / 2)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Power')
    axs[0].grid(True)

    axs[1].plot(freq, spect, 'blue')
    axs[1].plot(freq, np.power(np.abs(fftshift(fft(signal_prop))), 2), 'red')
    # axs[1].set_xlim(-np.pi / t_span * nt / (2 * np.pi), np.pi / t_span * nt / (2 * np.pi))
    axs[1].set_xlim(-1, 1)
    axs[1].set_xlabel('Normalized Frequency')
    axs[1].set_ylabel('Spectral Power')
    axs[1].grid(True)

    fig.show()

    fig2, axs2 = plt.subplots(3, 1)
    axs2[0].plot(t, np.power(np.absolute(signal_prop), 2), 'blue')
    axs2[0].plot(t, np.power(np.absolute(signal_end), 2), 'red')
    axs2[0].set_xlim(-t_span / 2, t_span / 2)
    axs2[0].set_xlabel('Time')
    axs2[0].set_ylabel('Power')
    axs2[0].grid(True)

    axs2[1].plot(freq, np.power(np.abs(fftshift(fft(signal_prop))), 2), 'blue')
    axs2[1].plot(freq, np.power(np.abs(fftshift(fft(signal_end))), 2), 'red')
    # axs2[1].set_xlim(-np.pi / t_span * nt / (2 * np.pi), np.pi / t_span * nt / (2 * np.pi))
    axs2[1].set_xlim(-5, 5)
    axs2[1].set_xlabel('Normalized Frequency')
    axs2[1].set_ylabel('Spectral Power')
    axs2[1].grid(True)

    axs2[2].plot(t, np.absolute(signal_prop - signal_end), 'blue')
    axs2[2].set_xlim(-t_span / 2, t_span / 2)
    axs2[2].set_xlabel('Time')
    axs2[2].set_ylabel('Power')
    axs2[2].set_yscale('log')
    axs2[2].grid(True)

    fig2.show()

    print(-np.pi / t_span * nt / (2 * np.pi), freq[0])
    print(check_energy(signal, t_span, fft(signal)))
    print("Energy diff: ", abs(energy_init - energy_end))
