import numpy as np
from scipy import interpolate
import time


def generate_excitation(f0,
                        no_of_harmonics,
                        harmonics_amplitudes,
                        samples,
                        fs,
                        jitter_factor,
                        sweep_params,
                        rng=None,
                        complex_output=False):

    if rng is None:
        rng = np.random.default_rng()

    excitation_length = len(samples)
    phase_0 = rng.random(no_of_harmonics) * 2 * np.pi  # initial phases

    #  jitter
    l_jitter = np.ceil(fs/f0)
    n_jitter = np.ceil(excitation_length/l_jitter) + 1
    df0_jitter_base = jitter_factor * f0 * (2*rng.random(int(n_jitter))-1)
    t_jitter_base = np.arange(n_jitter)*l_jitter/fs

    #  sweep
    alpha = sweep_params[2]
    df_sign = np.sign(sweep_params[1]-sweep_params[0])
    df = np.abs(sweep_params[1]-sweep_params[0])
    df = df*(1+alpha)
    if df_sign >= 0:
        f0_sweep = sweep_params[0] + df * (-alpha + np.power(alpha, (samples+1)/excitation_length))
    else:
        f0_sweep = sweep_params[1] + df * (-alpha + np.power(alpha, (1 - ((samples+1)/excitation_length))))

    x_tmp = t_jitter_base.reshape(len(t_jitter_base), 1)
    y_tmp = df0_jitter_base.reshape(len(df0_jitter_base), 1)
    xx_tmp = samples/fs
    tck = interpolate.splrep(x_tmp, y_tmp)
    f0_jitter = interpolate.splev(xx_tmp, tck, der=0)
    f0_sweep_and_jitter = f0_sweep + f0_jitter
    omega_0_jitter = 2*np.pi*f0_sweep_and_jitter/fs

    y = np.zeros_like(samples)
    if harmonics_amplitudes.size == 0:
        harmonics_amplitudes = np.ones(no_of_harmonics) / no_of_harmonics

    for i in range(no_of_harmonics):
        phase = ((i+1)*omega_0_jitter).cumsum()
        if complex_output == True:
            y = y + harmonics_amplitudes[i] * np.exp(1j*(phase+phase_0[i]))
        else:
            y = y + harmonics_amplitudes[i] * np.cos(phase+phase_0[i])

    return y, f0_sweep_and_jitter
