from timeit import default_timer as timer
import logging

from data_processor.dataset.ife_dataset import IFEDataset
from vowel_synthesis.vowels_synthesizer import vowels_synthesizer
from utils.AFE.process_audio import process_audio


# _, F0_all, FB_params, FB_data = vs.generate_segments(filename_core, SNR_dB, no_of_segments, segment_length, silence_length, F0_min, F0_max, Fs)

def test_AFE():
    # if filename_core is not None:
    #     filename_raw = filename_core + ".wav"
    #     F0_filename_out = filename_core + "_F0.wav"
    #     filename_out = filename_core + f"_snr_{SNR_dB}.wav"
    # else:
    #     filename_raw = None
    #     F0_filename_out = None
    #     filename_out = None

    # filename_wav = "d:/M_Blok/GitHub/HSI_data/training/Interspeech_2021/15000events_50-400Hz_length500ms_gaps0ms.wav"
    filename_wav = "d:/M_Blok/GitHub/HSI_data/training/testowe/15000events_50-400Hz_length500ms_gaps0ms.wav"

    logging.info('START:')
    # start = timer()
    # y_all, F0_all = vowels_generation(filename_raw, F0_filename_out, no_of_segments=no_of_segments, 
    #                                     fs = Fs, segment_length = segment_length, silence_length = silence_length, f0_min=F0_min, f0_max=F0_max, rng=self.rng) # filename == None if wave file should not be saved
    # end = timer()
    # logging.info(f"vowels_generation elapsed time:{(end - start):.2f}") # Time in seconds, e.g. 5.38091952400282

    # start = timer()
    # y_noise = add_noise(y_all, filename_out, SNR_dB, normalization=None, rng=self.rng) # TODO y_all can be replaced with filename 
    # # y_noise = add_noise(filename_raw, filename_out, SNR_dB) # TODO allow filename_out = None
    # end = timer()
    # logging.info(f"add_noise ({SNR_dB:.1f}dB) elapsed time:{(end - start):.2f}") # Time in seconds, e.g. 5.38091952400282

    start = timer()
    AFE_cfg = {"filter_bank_mode": "ConstQ", 
                "filter_bank_smoothing_mode": "delay",
                "CMPO_smoothing_mode": "none"}    
    FB_params, FB_data, _ = process_audio(filename_wav, no_of_filters = 20, AFE_cfg = AFE_cfg)
    end = timer()
    logging.info(f"process_audio elapsed time:{(end - start):.2f}") # Time in seconds, e.g. 5.38091952400282
    # logging.info(FB_params)
    # logging.info(FB_CMPO)

    

test_AFE()