from vowel_synthesis.main_generation_procedure import vowels_generation, add_noise
from utils.AFE.process_audio import process_audio
import logging
from pathlib import Path
import json

from timeit import default_timer as timer
import numpy as np
import time

class vowels_synthesizer:
    rng = None
    rng_seed = None
    def __init__(self, sequence_index = 0, rng_state_filename = None):
        # https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
        # np.random.seed(seed)

        if rng_state_filename is not None:
            if not Path(rng_state_filename).is_file():
                sq1 = np.random.SeedSequence()
                rng_state = sq1.entropy

                # save state
                Path(rng_state_filename).parent.mkdir(parents=True, exist_ok=True)
                with open(rng_state_filename,'w+') as f:
                    json.dump(rng_state,f)            

            # force recreation of rng and use of the saved state
            self.rng = None

        rng_seed = -1
        if self.rng is None:
            if rng_state_filename is not None:
                if Path(rng_state_filename).is_file():
                    # load state
                    with open(rng_state_filename,'r+') as f:
                        rng_state = json.load(f)
                    sq1 = np.random.SeedSequence(rng_state)
                else:
                    sq1 = np.random.SeedSequence()
            else:
                sq1 = np.random.SeedSequence()
            rng_state = sq1.entropy
            logging.info(f"rng_state: {rng_state}")
            logging.info(f"rng_sequence_index: {sequence_index}")

            rng_seeds = sq1.generate_state(sequence_index+1)
            rng_seed = rng_seeds[-1]
            logging.info(f"rng_seed: {rng_seed}")
            self.rng = np.random.default_rng(rng_seed)
        
        self.rng_seed = rng_seed


    # def get_segments(self, no_of_segments, SNR_dB):
    #     y_all, F0_all, FB_params, FB_CMPO = self.generate_segments(None, SNR_dB=SNR_dB, no_of_segments=no_of_segments)
    #     return y_all, F0_all, FB_params, FB_CMPO

    def generate_segments(self, filename_core, SNR_dB, no_of_segments=1000, segment_length = 500, silence_length = 0, F0_min = 50, F0_max = 400, Fs = 8000, no_of_filters = 20, AFE_cfg = None):
        if filename_core is not None:
            filename_raw = filename_core + ".wav"
            F0_filename_out = filename_core + "_F0.dat"
            filename_out = filename_core + f"_snr_{SNR_dB}.wav"
        else:
            filename_raw = None
            F0_filename_out = None
            filename_out = None

        if filename_core is not None:
            logging.info(f"filename_core:{filename_core}")
        logging.info('START:')
        start = timer()
        y_all, F0_all = vowels_generation(filename_raw, F0_filename_out, no_of_segments=no_of_segments, 
                                          fs = Fs, segment_length = segment_length, silence_length = silence_length, f0_min=F0_min, f0_max=F0_max, rng=self.rng) # filename == None if wave file should not be saved
        end = timer()
        logging.info(f"vowels_generation elapsed time:{(end - start):.2f}") # Time in seconds, e.g. 5.38091952400282

        start = timer()
        y_noise = add_noise(y_all, filename_out, SNR_dB, normalization=None, rng=self.rng) # TODO y_all can be replaced with filename 
        # y_noise = add_noise(filename_raw, filename_out, SNR_dB) # TODO allow filename_out = None
        end = timer()
        logging.info(f"add_noise ({SNR_dB:.1f}dB) elapsed time:{(end - start):.2f}") # Time in seconds, e.g. 5.38091952400282

        start = timer()
        FB_params, FB_data, _ = process_audio(filename_in = None, x = y_noise, Fs=Fs, no_of_filters = no_of_filters, AFE_cfg = AFE_cfg)
        end = timer()
        logging.info(f"process_audio elapsed time:{(end - start):.2f}") # Time in seconds, e.g. 5.38091952400282
        # logging.info(FB_params)
        # logging.info(FB_CMPO)

        return y_all, F0_all, FB_params, FB_data

def test(no_of_segments = None, rng_state_filename = None):
    vs = vowels_synthesizer(0, rng_state_filename)

    filename_core = "./test2" 
    SNR_dB = 40
    if no_of_segments is None:
        no_of_segments = 3000

    vs.generate_segments(filename_core, SNR_dB, no_of_segments)