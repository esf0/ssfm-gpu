import tensorflow as tf
import numpy as np

from tensorflow._api.v2.signal import fft, fftshift, ifft, ifftshift
from tensorflow import cast

# methods for Schrödinger equation (two polarisations)


def tf_ssfm_dispersive_step(signal, dispersion):

    return ifft(fft(signal) * dispersion)


def tf_ssfm_nonlinear_step(signal, gamma, delta_z):

    abs_signal = cast(tf.math.abs(signal), tf.complex128)
    return signal * tf.math.exp(cast(1.0j * delta_z * gamma, tf.complex128) * abs_signal * abs_signal)


def tf_fiber_propogate(initial_signal, t_span, fiber_length, n_span, gamma, beta2, alpha=0, beta3=0):

    if abs(fiber_length) < 1e-15:
        return initial_signal

    dz = fiber_length / n_span

    n = len(initial_signal)
    w = fftshift(np.array([(i - n / 2) * (2. * np.pi / t_span) for i in range(n)], dtype=complex))
    w2 = tf.math.pow(w, 2)
    w3 = tf.math.pow(w, 3)

    dispersion = tf.math.exp((0.5j * beta2 * w2 + 1. / 6. * beta3 * w3 - alpha / 2.) * dz)
    dispersion_half = tf.math.exp((0.5j * beta2 * w2 + 1. / 6. * beta3 * w3 - alpha / 2.) * dz / 2.)
    dispersion_mhalf = tf.math.exp((0.5j * beta2 * w2 + 1. / 6. * beta3 * w3 - alpha / 2.) * -dz / 2.)

    # D/2
    signal = tf_ssfm_dispersive_step(initial_signal, dispersion_half)

    for n in range(n_span):
        signal = tf_ssfm_nonlinear_step(signal, gamma, dz)
        signal = tf_ssfm_dispersive_step(signal, dispersion)

    # -D/2
    signal = tf_ssfm_dispersive_step(signal, dispersion_mhalf)

    return signal


def propagate_schrodinger(channel, signal, sample_freq):
    # schrodinger
    dt = 1 / sample_freq
    nt = len(signal)
    # print(nt)
    t_span = dt * nt
    # start_time = datetime.now()

    sq_gain = tf.cast(tf.math.sqrt(channel['gain']), tf.complex128)
    std = tf.cast(tf.math.sqrt(channel['noise_density'] * sample_freq), tf.complex128)
    # one_over_sq_2 = tf.cast(1. / tf.math.sqrt(2.), tf.complex128)

    for span_ind in range(channel['n_spans']):

        signal = tf_fiber_propogate(signal, t_span,
                                    channel['z_span'],
                                    channel['nz'],
                                    channel['gamma'],
                                    channel['beta2'],
                                    alpha=channel['alpha'],
                                    beta3=channel['beta3'])

        noise = tf.complex(tf.random.normal([nt], 0, 1, dtype=tf.float64),
                           tf.random.normal([nt], 0, 1, dtype=tf.float64))

        signal = sq_gain * signal + noise * std

    # end_time = datetime.now()
    # time_diff = (end_time - start_time)
    # execution_time = time_diff.total_seconds() * 1000
    # print("Signal propagation took", execution_time, "ms")

    return signal


def dispersion_compensation(channel, signal, dt):
    """
    Compensate dispersion.

    Args:
        channel: dictionary with channel specification
        signal: signal
        dt: time step for signal (1 / sample_frequency)

    Returns:
        (tuple): tuple containing:

            signal_cdc (tf_tensor): signal with compensated dispersion

    """

    #  Dispersion compensation #
    nt_cdc = len(signal)
    t_span = nt_cdc * dt
    w = fftshift(np.array([(i - nt_cdc / 2) * (2. * np.pi / t_span) for i in range(nt_cdc)], dtype=complex))
    w2 = tf.math.pow(w, 2)
    w3 = tf.math.pow(w, 3)
    dispersion = tf.math.exp((0.5j * channel['beta2'] * w2 + 1. / 6. * channel['beta3'] * w3) *
                             (-channel['z_span'] * channel['n_spans']))
    signal_cdc = tf_ssfm_dispersive_step(tf.cast(signal, tf.complex128), dispersion)

    return signal_cdc


# methods for Manakov equation (two polarisations)


def tf_ssfm_manakov_dispersive_step(first, second, dispersion):

    first_new = ifft(fft(first) * dispersion)
    second_new = ifft(fft(second) * dispersion)
    return first_new, second_new


def tf_ssfm_manakov_nonlinear_step(first, second, gamma, delta_z):

    abs_first = cast(tf.math.abs(first), tf.complex128)
    abs_second = cast(tf.math.abs(second), tf.complex128)
    first_new = first * tf.math.exp(cast(1.0j * delta_z * 8.0 / 9.0 * gamma, tf.complex128) * (abs_first * abs_first + abs_second * abs_second))
    second_new = second * tf.math.exp(cast(1.0j * delta_z * 8.0 / 9.0 * gamma, tf.complex128) * (abs_first * abs_first + abs_second * abs_second))
    return first_new, second_new


def tf_manakov_fiber_propogate(initial_first, initial_second, t_span, fiber_length, n_steps, gamma, beta2, alpha=0, beta3=0):

    if abs(fiber_length) < 1e-15:
        return initial_first, initial_second

    dz = fiber_length / n_steps

    if len(initial_first) != len(initial_second):
        print('[tf_manakov_fiber_propogate] Error: sizes of first and second polarisation have to be the same!')
        return initial_first, initial_second

    n = len(initial_first)
    w = tf.signal.fftshift(np.array([(i - n / 2) * (2. * np.pi / t_span) for i in range(n)], dtype=complex))
    w2 = tf.math.pow(w, 2)
    w3 = tf.math.pow(w, 3)

    dispersion = tf.math.exp((0.5j * beta2 * w2 + 1. / 6. * beta3 * w3 - alpha / 2.) * dz)
    dispersion_half = tf.math.exp((0.5j * beta2 * w2 + 1. / 6. * beta3 * w3 - alpha / 2.) * dz / 2.)
    dispersion_mhalf = tf.math.exp((0.5j * beta2 * w2 + 1. / 6. * beta3 * w3 - alpha / 2.) * -dz / 2.)

    # D/2
    first, second = tf_ssfm_manakov_dispersive_step(initial_first, initial_second, dispersion_half)

    for n in range(n_steps):
        first, second = tf_ssfm_manakov_nonlinear_step(first, second, gamma, dz)
        first, second = tf_ssfm_manakov_dispersive_step(first, second, dispersion)

    # -D/2
    first, second = tf_ssfm_manakov_dispersive_step(first, second, dispersion_mhalf)

    return first, second


def propagate_manakov(channel, signal_x, signal_y, sample_freq):

    dt = 1 / sample_freq
    nt = len(signal_x)
    # print(nt)
    t_span = dt * nt
    # start_time = datetime.now()

    sq_gain = tf.cast(tf.math.sqrt(channel['gain']), tf.complex128)
    std = tf.cast(tf.math.sqrt(channel['noise_density'] * sample_freq), tf.complex128)
    one_over_sq_2 = tf.cast(1. / tf.math.sqrt(2.), tf.complex128)

    for span_ind in range(channel['n_spans']):
        signal_x, signal_y = tf_manakov_fiber_propogate(signal_x, signal_y, t_span,
                                                        channel['z_span'],
                                                        channel['nz'],
                                                        channel['gamma'],
                                                        channel['beta2'],
                                                        alpha=channel['alpha'],
                                                        beta3=channel['beta3'])
        #
        # noise_x = (np.random.normal(0, 1, size=nt) + 1.0j * np.random.normal(0, 1, size=nt)) * one_over_sq_2
        # noise_y = (np.random.normal(0, 1, size=nt) + 1.0j * np.random.normal(0, 1, size=nt)) * one_over_sq_2

        noise_x = tf.complex(tf.random.normal([nt], 0, 1, dtype=tf.float64), tf.random.normal([nt], 0, 1, dtype=tf.float64)) * one_over_sq_2
        noise_y = tf.complex(tf.random.normal([nt], 0, 1, dtype=tf.float64), tf.random.normal([nt], 0, 1, dtype=tf.float64)) * one_over_sq_2

        signal_x = sq_gain * signal_x + noise_x * std
        signal_y = sq_gain * signal_y + noise_y * std

    # end_time = datetime.now()
    # time_diff = (end_time - start_time)
    # execution_time = time_diff.total_seconds() * 1000
    # print("Signal propagation took", execution_time, "ms")

    return signal_x, signal_y


def propagate_manakov_backward(channel, signal_x, signal_y, sample_freq):

    dt = 1 / sample_freq
    nt = len(signal_x)
    # print(nt)
    t_span = dt * nt
    # start_time = datetime.now()

    sq_gain = tf.cast(tf.math.sqrt(channel['gain']), tf.complex128)
    std = tf.cast(tf.math.sqrt(channel['noise_density'] * sample_freq), tf.complex128)
    one_over_sq_2 = tf.cast(1. / tf.math.sqrt(2.), tf.complex128)

    for span_ind in range(channel['n_spans']):
        noise_x = tf.complex(tf.random.normal([nt], 0, 1, dtype=tf.float64),
                             tf.random.normal([nt], 0, 1, dtype=tf.float64)) * one_over_sq_2
        noise_y = tf.complex(tf.random.normal([nt], 0, 1, dtype=tf.float64),
                             tf.random.normal([nt], 0, 1, dtype=tf.float64)) * one_over_sq_2

        signal_x = (signal_x + noise_x * std) / sq_gain
        signal_y = (signal_y + noise_y * std) / sq_gain

        signal_x, signal_y = tf_manakov_fiber_propogate(signal_x, signal_y, t_span,
                                                        channel['z_span'],
                                                        channel['nz'],
                                                        channel['gamma'],
                                                        channel['beta2'],
                                                        alpha=channel['alpha'],
                                                        beta3=channel['beta3'])
        #
        # noise_x = (np.random.normal(0, 1, size=nt) + 1.0j * np.random.normal(0, 1, size=nt)) * one_over_sq_2
        # noise_y = (np.random.normal(0, 1, size=nt) + 1.0j * np.random.normal(0, 1, size=nt)) * one_over_sq_2



    # end_time = datetime.now()
    # time_diff = (end_time - start_time)
    # execution_time = time_diff.total_seconds() * 1000
    # print("Signal propagation took", execution_time, "ms")

    return signal_x, signal_y


def dispersion_compensation_manakov(channel, signal_x, signal_y, dt):
    """
    Compensate dispersion.

    Args:
        channel: dictionary with channel specification
        signal_x: signal on the first (x) polarisation
        signal_y: signal on the second (y) polarisation
        dt: time step for signal (1 / sample_frequency)

    Returns:
        (tuple): tuple containing:

            signal_cdc_x (tf_tensor): signal on the first (x) polarisation with compensated dispersion
            signal_cdc_y (tf_tensor): signal on the second (y) polarisation with compensated dispersion

    """

    #  Dispersion compensation #
    nt_cdc = len(signal_x)
    t_span = nt_cdc * dt
    w = fftshift(np.array([(i - nt_cdc / 2) * (2. * np.pi / t_span) for i in range(nt_cdc)], dtype=complex))
    w2 = tf.math.pow(w, 2)
    w3 = tf.math.pow(w, 3)
    dispersion = tf.math.exp((0.5j * channel['beta2'] * w2 + 1. / 6. * channel['beta3'] * w3) *
                             (-channel['z_span'] * channel['n_spans']))
    signal_cdc_x, signal_cdc_y = tf_ssfm_manakov_dispersive_step(tf.cast(signal_x, tf.complex128),
                                                                 tf.cast(signal_y, tf.complex128),
                                                                 dispersion)

    return signal_cdc_x, signal_cdc_y


# Channel parameters


def get_default_channel_parameters():
    """
    Get default optical channel parameters.

    Returns:
        Dictionary with default optical channel parameters
            - 'n_spans' -- Total number of spans
            - 'z_span' -- Length of each span in [km]
            - 'alpha_db' -- :math:`\\alpha_{dB}`
            - 'alpha' -- :math:`\\alpha`
            - 'gamma' -- :math:`\\gamma`
            - 'noise_figure_db' -- :math:`NF_{dB}`
            - 'noise_figure' -- :math:`NF`
            - 'gain' -- :math:`G`
            - 'dispersion_parameter' -- :math:`D`
            - 'beta2' -- :math:`\\beta_2`
            - 'beta3' -- :math:`\\beta_3`
            - 'h_planck' -- Planck constant
            - 'fc' -- Carrier frequency math:`f_{carrier}`
            - 'dz' -- Fixed spatial step in [km]
            - 'nz' -- Number of steps per each spatial span
            - 'noise_density' -- Noise density math:`h \\cdot f_{carrier} \\cdot (G - 1) \\cdot NF`

    """

    channel = {}
    channel['n_spans'] = 12  # Number of spans
    channel['z_span'] = 80  # Span Length [km]
    channel['alpha_db'] = 0.225  # Attenuation coefficient [dB km^-1]
    channel['alpha'] = channel['alpha_db'] / (10 * np.log10(np.exp(1)))
    channel['gamma'] = 1.2  # Non-linear Coefficient [W^-1 km^-1]. Default = 1.2
    channel['noise_figure_db'] = 4.5  # Noise Figure [dB]. Default = 4.5
    channel['noise_figure'] = 10 ** (channel['noise_figure_db'] / 10)
    channel['gain'] = np.exp(channel['alpha'] * channel['z_span']) # gain for one span
    channel['dispersion_parameter'] = 16.8 #  [ps nm^-1 km^-1]  dispersion parameter
    channel['beta2'] = -(1550e-9 ** 2) * (channel['dispersion_parameter'] * 1e-3) / (2 * np.pi * 3e8)  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    channel['beta3'] = 0
    channel['h_planck'] = 6.62607015e-34  # Planck's constant [J/s]
    channel['fc'] = 299792458 / 1550e-9  # carrier frequency
    channel['dz'] = 1.0  # length of the step for SSFM [km]
    channel['nz'] = int(channel['z_span'] / channel['dz'])  # number of steps per each span
    channel['noise_density'] = channel['h_planck'] * channel['fc'] * (channel['gain'] - 1) * channel['noise_figure']
    channel['seed'] = 'fixed'

    return channel


def create_channel_parameters(n_spans, z_span, alpha_db, gamma, noise_figure_db, dispersion_parameter, dz, seed='fixed'):

    alpha = alpha_db / (10 * np.log10(np.exp(1)))
    noise_figure = 10 ** (noise_figure_db / 10)
    gain = np.exp(alpha * z_span)  # gain for one span
    beta2 = -(1550e-9 ** 2) * (dispersion_parameter * 1e-3) / (2 * np.pi * 3e8)  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    beta3 = 0
    h_planck = 6.6256e-34  # Planck's constant [J/s]
    # nu = 299792458 / 1550e-9  # light frequency carrier [Hz]
    fc = 299792458 / 1550e-9  # carrier frequency
    nz = int(z_span / dz)  # number of steps per each span
    noise_density = h_planck * fc * (gain - 1) * noise_figure

    channel = {}
    channel['n_spans'] = n_spans  # Number of spans
    channel['z_span'] = z_span  # Span Length [km]
    channel['alpha_db'] = alpha_db  # Attenuation coefficient [dB km^-1]
    channel['alpha'] = alpha
    channel['gamma'] = gamma  # Non-linear Coefficient [W^-1 km^-1]. Default = 1.2
    channel['noise_figure_db'] = noise_figure_db  # Noise Figure [dB]. Default = 4.5
    channel['noise_figure'] = noise_figure
    channel['gain'] = gain  # gain for one span
    channel['dispersion_parameter'] = dispersion_parameter  # [ps nm^-1 km^-1]  dispersion parameter
    channel['beta2'] = beta2  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    channel['beta3'] = beta3
    channel['h_planck'] = h_planck  # Planck's constant [J/s]
    channel['fc'] = h_planck  # carrier frequency
    channel['dz'] = dz  # length of the step for SSFM [km]
    channel['nz'] = nz  # number of steps per each span
    channel['noise_density'] = noise_density
    channel['seed'] = seed

    return channel