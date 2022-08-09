from vowel_synthesis.generate_excitation import generate_excitation
import numpy as np
from scipy import interpolate
from scipy.io.wavfile import write as audiowrite
# import time
from tqdm import tqdm


def vowels_generation(filename_out = None, F0_filename_out = None, fs = 8000, no_of_segments = 10, segment_length = 500, silence_length = 0, f0_min=50, f0_max=400, rng=None):
    # fs = 8000  # Hz
    # no_of_segments = 10
    # segment_length = 500  # ms
    # silence_length = 0  # ms of silence between segments
    # f0_min = 50  # Hz
    # f0_max = 400  # Hz

    if rng is None:
        rng = np.random.default_rng()

    f0_mode = "linear"
    f0_jitter_range = [0.005, 0.01]
    intonation_range = 100  # Hz
    intonation_bound = intonation_range/2
    sweep_factor = 0.1
    segment_type = "poly"
    ## n_db = [-60]
    # y_all = np.empty(0)
    # f0_all = np.empty(0)
    # t_all = np.empty(0)
    f0_vanilla = np.empty(0)
    # f0_to_sweep = np.empty(0)

    no_of_samples_in_segment = round(segment_length/1000*fs)
    n_samples = np.arange(no_of_samples_in_segment)
    if silence_length > 0:
        no_of_samples_silence = round(silence_length/1000*fs)
        n_samples_silence = np.arange(no_of_samples_silence)
        silence = np.zeros_like(n_samples_silence)
        silence_nans = np.zeros_like(silence)
        silence_nans[:] = np.nan
    else:
        no_of_samples_silence = 0

    y_all = np.empty((no_of_segments*(no_of_samples_in_segment+no_of_samples_silence),), dtype=np.float32)
    f0_all = np.empty((no_of_segments*(no_of_samples_in_segment+no_of_samples_silence),), dtype=np.float32)
    t_all = np.empty((no_of_segments*(no_of_samples_in_segment+no_of_samples_silence),), dtype=np.float32)

    with tqdm(total=no_of_segments, unit=" segments", mininterval=0.1, dynamic_ncols=True) as progressbar:
        n_offset = 0
        for i in range(no_of_segments):
            f0 = (f0_min + rng.random(1)*(f0_max-intonation_bound-f0_min+intonation_bound))[0]
            f0_vanilla = np.append(f0_vanilla, f0)
            jitter_factor = f0_jitter_range[0] + rng.random(1)*(f0_jitter_range[1]-f0_jitter_range[0])
            intonator = rng.random(1)
            if intonator >= 0.5:
                f1 = f0
                f2 = intonation_range*(rng.random(1)-0.5)+f0
            else:
                f1 = intonation_range*(rng.random(1)-0.5)+f0
                f2 = f0
            if f1 > f0_max:
                f1 = f0_max
            if f1 < f0_min:
                f1 = f0_min
            if f2 > f0_max:
                f2 = f0_max
            if f2 < f0_min:
                f2 = f0_min
            sweep_params = [f1, f2, sweep_factor]
            no_of_harmonics = int(np.floor((fs/2)/max(f1, f2)))
            fk = (np.arange(no_of_harmonics)+1)*f0
            v = 0.15+rng.random(1)*0.25
            s = 1.4+rng.random(1)*1.1
            harmonics_amplitudes = np.power(np.sin(np.pi*(np.power(2*fk/fs, v))), np.power(s, 4))
            # no_of_samples_in_segment = round(segment_length/1000*fs)
            # samples = np.arange(no_of_samples_in_segment)
            # if silence_length > 0:
            #     no_of_samples_silence = round(silence_length/1000*fs)
            #     samples_silence = np.arange(no_of_samples_silence)
            #     silence = np.zeros_like(samples_silence)
            #     silence_nans = np.zeros_like(silence)
            #     silence_nans[:] = np.nan

            [y, f0_sweep_and_jitter_seg] = generate_excitation(f0,
                                                            no_of_harmonics,
                                                            harmonics_amplitudes,
                                                            n_samples,
                                                            fs,
                                                            jitter_factor,
                                                            sweep_params,
                                                            rng=rng)

            t_transition = 15 + rng.random(1)*(segment_length/10)
            n_transition = int((np.ceil(t_transition/1000*fs))[0])
            window_shape = np.append(np.linspace(0, 1, num=n_transition), np.ones((1, no_of_samples_in_segment - 3*n_transition)))
            window_shape = np.append(window_shape, np.linspace(1, 0, num=2*n_transition))
            window = np.sin(np.pi/2*window_shape)
            y = y*window
            a = 0.1 + 0.9*rng.random(1)
            y = a * y / max(abs(y))

            if silence_length > 0:
                y_seg = np.append(y, silence)
                y_seg_length = len(y_seg)
                f0_seg = np.append(f0_sweep_and_jitter_seg, silence_nans)
            else:
                y_seg = y
                y_seg_length = len(y_seg)
                f0_seg = f0_sweep_and_jitter_seg
            t_seg = np.linspace(0, y_seg_length - 1, y_seg_length) / fs


            # y_all = np.append(y_all, y_seg)
            # f0_all = np.append(f0_all, f0_seg)
            # if t_all.size == 0:
            #     t_all = t_seg
            # else:
            #     t_all = np.append(t_all, (t_all[-1] + 1/fs + t_seg))
            next_n_offset = n_offset + (no_of_samples_in_segment+no_of_samples_silence)
            y_all[n_offset:next_n_offset] = y_seg
            f0_all[n_offset:next_n_offset] = f0_seg
            if n_offset == 0:
                t_all[n_offset:next_n_offset] = t_seg
            else:
                t_all[n_offset:next_n_offset] = t_all[n_offset-1] + 1/fs + t_seg
            n_offset = next_n_offset
            progressbar.update(1)

    shimmer_factor = 1.5
    shimmer_factor_db = 20*np.log10(shimmer_factor)
    l_shimmer = int(np.ceil(0.01*fs))
    n = len(y_all)
    n_shimmer = int(np.ceil(n/l_shimmer))+1
    a_shimmer_base = np.power(10, shimmer_factor_db*(2*rng.random(n_shimmer)-1)/20, dtype=np.float32)
    t_shimmer_base = np.linspace(0, n_shimmer-1, n_shimmer, dtype=np.float32)*l_shimmer/fs

    x_tmp = t_shimmer_base.reshape(len(t_shimmer_base), 1)
    y_tmp = a_shimmer_base.reshape(len(a_shimmer_base), 1)
    xx_tmp = t_all
    tck = interpolate.splrep(x_tmp, y_tmp)
    a_shimmer = interpolate.splev(xx_tmp, tck, der=0).astype(np.float32)
    a_shimmer = a_shimmer / np.max(a_shimmer)

    y_all = np.multiply(y_all, a_shimmer)

    if filename_out is not None:
        audiowrite(filename_out, fs, (np.iinfo(np.int16).max*np.real(y_all)).astype(np.int16))

    if F0_filename_out is not None:
        with open(F0_filename_out, 'wb') as hf:
            f0_all.astype('float32').tofile(hf)

    return np.real(y_all), f0_all

def add_noise(y_in, filename_out, SNR_dB, fs = 8000, normalization=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    factor = np.power(10, -SNR_dB/20)

    signal_level = np.sqrt(np.mean(np.abs(y_in)**2))

    #noise = factor*signal_level*rng.randn(y_in.size)
    noise = factor*signal_level*np.array(rng.standard_normal(y_in.size),dtype=np.float32)

    # snr = np.floor(20*np.log10(signal_level/noise_level))
      
    y_all_n = y_in + noise

    if normalization == 'max':
        norm_factor = np.real(max(y_all_n))
        y_all_n = y_all_n/norm_factor

    if filename_out is not None: 
        audiowrite(filename_out, fs, (np.iinfo(np.int16).max*np.real(y_all_n)).astype(np.int16))
    return y_all_n
