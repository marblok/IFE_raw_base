from pathlib import PurePath, Path

from data_processor.dataset.ife_dataset import IFEDataset, IFEDataset_raw
from vowel_synthesis.vowels_synthesizer import vowels_synthesizer
from utils.AFE.process_audio import process_audio

import numpy as np
import logging

def create_data_from_file(data_path, dataset_options = None):
    """Creates dataset and data_loader for training or inference based on given data_path"""
    logging.info(f"Loading data from: {data_path}. This may take a while...")
    if "merged_data_filename" in data_path:
        F0_ref_path = None
        raw_data_path = None
        merged_data_path = data_path["PATH"] + data_path["merged_data_filename"]
    else:
        merged_data_path = None
        if "F0_ref_filename" in data_path:
            if data_path["F0_ref_filename"] is not None:
                F0_ref_path = Path(data_path["PATH"], data_path["F0_ref_filename"]).resolve().as_posix()
            else:
                F0_ref_path = None
        else:
            F0_ref_path = None

        WAVE_data_path = None
        if "WAV_filename" in data_path:
            if data_path["WAV_filename"] is not None:
                WAVE_data_path = Path(data_path["PATH"], data_path["WAV_filename"]).resolve().as_posix()
                raw_data_path = None

        if WAVE_data_path is None:
            raw_data_path = Path(data_path["PATH"], data_path["FB_data_filename"]).resolve().as_posix()
        else:
            raw_data_path = None

        # if ((raw_data_path is None) and (WAVE_data_path is None)) or (F0_ref_path is None):
        #     raise Exception("create_data_from_wav_file: (FB_data_filename and WAV_filename) or F0_ref_filename are not defined")
        if (raw_data_path is None) and (WAVE_data_path is None):
            raise Exception("create_data_from_wav_file: both FB_data_filename and WAV_filename are not defined")

    if WAVE_data_path is not None:
        # load wav file and process it
        if "AFE" not in dataset_options:
            AFE_cfg = {"filter_bank_mode": "ConstQ", 
                        "filter_bank_smoothing_mode": "delay",
                        "CMPO_smoothing_mode": "none",
                        "FB_to_CMPO_smoothing_factor": 1.0}    
            logging.info(r"Warning: dataset_options[\"AFE\"] does not exist, using default AFE config")
            dataset_options["AFE"] = AFE_cfg
        else:
            if "FB_to_CMPO_smoothing_factor" not in dataset_options["AFE"]:
                dataset_options["AFE"]["FB_to_CMPO_smoothing_factor"] = -1.0 # signal that FB_to_CMPO_smoothing_factor wasn't defined
                logging.info(r"Warning: dataset_options[\"AFE\"] does not contain \"FB_to_CMPO_smoothing_factor\" setting it to -1.0")

        FB_params, FB_data, _ = process_audio(filename_in = WAVE_data_path, no_of_filters = int(dataset_options["FB_data_no_of_inputs"]/2), AFE_cfg = dataset_options["AFE"])
        if F0_ref_path is None:
            dataset = IFEDataset_raw(
                FB_data=FB_data, FB_params=FB_params, dataset_options=dataset_options) # it should be also possible to feed min/max label with values_count file
        else:
            dataset = IFEDataset(
                F0_ref_input_file=F0_ref_path,
                FB_data=FB_data, FB_params=FB_params, dataset_options=dataset_options) # it should be also possible to feed min/max label with values_count file
    else:
        dataset = IFEDataset(merged_input_file=merged_data_path, 
            FB_data_input_file=raw_data_path, 
            F0_ref_input_file=F0_ref_path,
            dataset_options=dataset_options) # it should be also possible to feed min/max label with values_count file
    logging.info(r"IFEDataset created")

    return dataset

class SNR_rng_state:
    rng = None
    
    @staticmethod
    def init(rng_seed, SNR_index) -> None:
        SNR_rng_state.rng = np.random.default_rng(rng_seed)
        if SNR_index > 0:
            tmp = SNR_rng_state.rng.random(SNR_index)

def get_SNR_dB(rng_state_filename, SNR_index, SNR_dB_range, vs = None):
    if vs is None:
        vs  = vowels_synthesizer(0, rng_state_filename)

    if (SNR_rng_state.rng is None) or (SNR_index == 0):
        SNR_rng_state.init(vs.rng_seed, SNR_index)

    # randomize SNR from given range
    #if (type(SNR_dB_range) == np.float64) or (type(SNR_dB_range) == float):
    if not hasattr(SNR_dB_range, "__len__"):
        SNR_dB = np.float64(SNR_dB_range)
    else:
        if SNR_dB_range[0] == SNR_dB_range[1]:
            SNR_dB = SNR_dB_range[0]
        else:
            SNR_dB = (SNR_dB_range[0] + (SNR_dB_range[1]-SNR_dB_range[0]) * SNR_rng_state.rng.random(1))[0]
    logging.info(f"SNR_dB = {SNR_dB:.3f}")

    return SNR_dB   

def create_data_by_synthesis(SNR_dB_range, no_of_segments, segment_length, silence_length, F0_min, F0_max, Fs, rng_state_filename, rng_epoch_index, SNR_index, save_validation_audio_to_wav_path = None, dataset_options = None):
    """Creates dataset and data_loader for training or inference based on given data_path"""
    logging.info(f"Starting data synthesizer...")

    if "AFE" not in dataset_options:
        AFE_cfg = {"filter_bank_mode": "ConstQ", 
                    "filter_bank_smoothing_mode": "delay",
                    "CMPO_smoothing_mode": "none",
                    "FB_to_CMPO_smoothing_factor": 1.0}    
        logging.info(r"Warning: dataset_options[\"AFE\"] does not exist, using default AFE config")
        dataset_options["AFE"] = AFE_cfg
    else:
        if "FB_to_CMPO_smoothing_factor" not in dataset_options["AFE"]:
            dataset_options["AFE"]["FB_to_CMPO_smoothing_factor"] = -1.0 # signal that FB_to_CMPO_smoothing_factor wasn't defined
            logging.info(r"Warning: dataset_options[\"AFE\"] does not contain \"FB_to_CMPO_smoothing_factor\" setting it to -1.0")

    vs  = vowels_synthesizer(rng_epoch_index, rng_state_filename)
    # if SNR_index == 0:
    #     SNR_rng_state.init(vs.rng_seed)
    SNR_dB = get_SNR_dB(rng_state_filename, SNR_index, SNR_dB_range, vs)

    if save_validation_audio_to_wav_path is None:
        filename_core = None 
    else:
        filename_core = (save_validation_audio_to_wav_path /  "validation_synth_audio").as_posix()
    _, F0_all, FB_params, FB_data = vs.generate_segments(filename_core, SNR_dB, no_of_segments, segment_length, silence_length, F0_min, F0_max, Fs, int(dataset_options["FB_data_no_of_inputs"]/2), dataset_options["AFE"])

    dataset = IFEDataset(F0_ref=F0_all, FB_data=FB_data, FB_params=FB_params, dataset_options=dataset_options, SNR=SNR_dB) # it should be also possible to feed min/max label with values_count file
    logging.info(r"IFEDataset created")

    return dataset