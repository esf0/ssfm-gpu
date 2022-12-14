import numpy as np

# convert NLSE equation
# dimensionless form iq_z +- 1/2 q_tt + |q|^2 * q = 0
# dimension form iQ_Z - beta2 / 2 Q_TT + gamma |Q|^2 * Q = 0
#
# T = T_0 * t
# Z = L * z
# L = T_0 ^ 2 / |beta2|
# Q = C * q
# C = 1 / (gamma * L) ^ (1/2) = |beta2| ^ (1/2) / (gamma ^ (1/2) * T_0)

# Manakov equation
# dimensionless form
# i q1_z + q1_tt +- 2 * (|q1|^2 + |q2|^2) q1 = 0
# i q2_z + q2_tt +- 2 * (|q1|^2 + |q2|^2) q2 = 0
# -1/+1 -> defocus / focus
#
# dimension form
# i Q1_Z - beta2 / 2 * Q1_TT + 8/9 * gamma * (|Q1|^2 + |Q2|^2) * Q1 = 0
# i Q2_Z - beta2 / 2 * Q2_TT + 8/9 * gamma * (|Q1|^2 + |Q2|^2) * Q2 = 0
#
# T = T_0 * t
# Z = L * z
# L = 2 * T_0 ^ 2 / |beta2|
# Q = C * q
# [2 / (8/9 * gamma * L)] ^ (1/2) = |beta2| ^ (1/2) / [(8/9 * gamma * L) ^ (1/2) * T_0]

# to run dimensionless form of NLSE solver
# beta2 = -+ 1
# gamma = 1

# to run dimensionless form of Manakov solver
# beta2 = -2
# gamma = +- 2 * 9/8 -> "+" focus / "-" defocus


def get_convert_coefficients_nlse(beta2, gamma, t0):

    coefficients = {}
    coefficients['T0'] = t0
    coefficients['L'] = t0 * t0 / abs(beta2)
    coefficients['C'] = 1.0 / (gamma * coefficients['L']) ** 0.5

    return coefficients


def get_convert_coefficients_manakov(beta2, gamma, t0):
    coefficients = {}
    coefficients['T0'] = t0
    coefficients['L'] = 2 * t0 * t0 / abs(beta2)
    coefficients['C'] = (2.0 / (8. / 9. * gamma * coefficients['L'])) ** 0.5

    return coefficients


def convert_forward(q, t, z, beta2, gamma, t0, type='nlse'):
    # convert from dimensionless to dimension form
    result = {}
    if type == 'nlse':
        coefficients = get_convert_coefficients_nlse(beta2, gamma, t0)
        result['Q'] = coefficients['C'] * q
    else:
        # manakov
        coefficients = get_convert_coefficients_manakov(beta2, gamma, t0)
        result['Q1'] = coefficients['C'] * q[0]
        result['Q2'] = coefficients['C'] * q[1]

    result['T'] = coefficients['T0'] * t
    result['Z'] = coefficients['L'] * z

    return result


def convert_inverse(q, t, z, beta2, gamma, t0, type='nlse'):
    # convert from dimension to dimensionless form
    result = {}
    if type == 'nlse':
        coefficients = get_convert_coefficients_nlse(beta2, gamma, t0)
        result['q'] = q / coefficients['C']
    else:
        # manakov
        coefficients = get_convert_coefficients_manakov(beta2, gamma, t0)
        result['q1'] = q[0] / coefficients['C']
        result['q2'] = q[1] / coefficients['C']

    result['t'] = t / coefficients['T0']
    result['z'] = z / coefficients['L']

    return result
