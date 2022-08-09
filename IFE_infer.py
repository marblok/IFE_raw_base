from joblib import PrintTime
from networks.MLP_2_layers import Model_auxdata, MLP_2_layers, MLP_multilayer, MLP_3_layers
from engine.inferer import infer_model
import sys
import json

from torch import torch
import numpy as np
from data_processor.creators import create_data_from_file, create_data_by_synthesis
from utils.constants import ModelBinaryAttributes
from data_processor.labels_transformer import LabelsTransformer

from pathlib import PurePath, Path
import logging
from utils.log_config import LogConfig
import traceback
import os

def get_model_folders(model_path, folder_prefix, folder_name):
    if len(folder_name) == 0:
        subfolders = [ f.path for f in os.scandir(model_path) if (f.name.startswith(folder_prefix)) and (f.is_dir()) ]
    else:
        subfolders = [model_path + folder_prefix + folder_name]
    return subfolders

def get_input_files(wav_file_path):
    """
    returns None if file or folder wav_file_path does not exist
    or list of tuples: (main folder, subfolder string, filenames)
    where filenames is a list of wav filenames in given subfolder of main folder
    """
    wav_file_path = Path(wav_file_path)

    if not wav_file_path.exists():
        # return file or folder does not exist
        return None

    if wav_file_path.is_file():
        # return [ wav_file_path.as_posix() ]
        wav_file_path = wav_file_path.absolute()

        # return: main folder, subfolder string, filename
        return [ (wav_file_path.parent, PurePath("."), [ wav_file_path.name ]) ]

    files = None
    if wav_file_path.is_dir():
        main_folder = wav_file_path.absolute()

        # find *.wav files in current directory (os.walk returns also the top folder)
        # find subfolders in current directory => collect info from those directories
        subfolders = [x[0] for x in os.walk(main_folder)]

        if len(subfolders) > 0:
            # find *.wav files in each subfolder of current directory
            files = []
            for subfolder in subfolders:
                subpath = PurePath(subfolder).relative_to(main_folder)
                # subfolder_path = Path(wav_file_path, subpath)
                filenames = [file.name for file in Path(subfolder).glob("*.wav")]
                files = files + [ (main_folder, subpath, filenames) ]

    return files

def IFE_infer(input_wav_files, model_path, output_data_path, log_cfg = None, output_file_format = "npz"):
    do_clear_log = False
    if log_cfg is None:
        log_cfg = LogConfig()
        log_cfg.init_logging()
        do_clear_log = True

    # load model first model (next epochs should be compatible)
    checkpoint_dataset_options = None
    checkpoint_label_transformer_options = None
    checkpoint_trainer_config = None
    # Load model state and configuration
    checkpoint = torch.load(PurePath(model_path).as_posix())

    model_config = None
    labels_transformer_options = None
    # extract a configuration subset used for particular model
    if ModelBinaryAttributes.DATASET_OPTIONS_OLD in checkpoint:
        checkpoint_dataset_options = checkpoint[ModelBinaryAttributes.DATASET_OPTIONS_OLD]
    if ModelBinaryAttributes.DATASET_OPTIONS in checkpoint:
        checkpoint_dataset_options = checkpoint[ModelBinaryAttributes.DATASET_OPTIONS]
    if ModelBinaryAttributes.LABEL_TRANSFORMER_OPTIONS in checkpoint:
        checkpoint_label_transformer_options = checkpoint[ModelBinaryAttributes.LABEL_TRANSFORMER_OPTIONS]
        labels_transformer_options = checkpoint_label_transformer_options
    if ModelBinaryAttributes.TRAINER_CONFIG in checkpoint:
        checkpoint_trainer_config = checkpoint[ModelBinaryAttributes.TRAINER_CONFIG]
    if ModelBinaryAttributes.MODEL_CONFIG in checkpoint:
        model_config = checkpoint[ModelBinaryAttributes.MODEL_CONFIG]
    else:
        logging.info("No model_config in checkpoint")
        # use model_config from checkpoint
        if "model_config" in checkpoint_trainer_config:
            model_config = checkpoint_trainer_config["model_config"]
        else:
            logging.info("No model_config in checkpoint_trainer_config")

    if model_config is None:
        raise Exception("model_config could not be determined")

    dataset_options = checkpoint_dataset_options
    dataset_options["shuffle_data"] = False
    dataset_options["maintain data continuity"] = True

    if ModelBinaryAttributes.TRAINING_DATASET_OPTIONS in checkpoint:
        checkpoint_training_dataset_options = checkpoint[ModelBinaryAttributes.TRAINING_DATASET_OPTIONS]
    else:
        checkpoint_training_dataset_options = None

    if labels_transformer_options is None:
        raise Exception("labels_transformer_options could not be determined")

    inference_dataset = None


    model_epoch = checkpoint[ModelBinaryAttributes.EPOCH]
    epoch_str = "epoch_" + str(model_epoch)
    logging.info(f"Epoch: {epoch_str}")

    # log model parameter 
    logging.info(f"model_path: {model_path}")
    logging.info(f"model_config: {model_config}")
    logging.info(f"labels_transformer_options: {labels_transformer_options}")
    logging.info(f"checkpoint_trainer_config: {checkpoint_trainer_config}")
    logging.info(f"dataset_options: {dataset_options}")
    logging.info(f"training_dataset_options: {checkpoint_training_dataset_options}") # from first epoch

    # select and create model
    # TODO add parameters and other model types
    network_model = None
    if model_config["type"] == "MLP_2_layers":
        network_model = MLP_2_layers(
            n_inputs= model_config["no_of_inputs"],
            n_classes= model_config["no_of_classes"],
            use_F0_too_low_class = labels_transformer_options["use_F0_too_low_class"]) # creation of model topology
    elif model_config["type"] == "MLP_multilayer":
        # TODO the class name should be changed
        network_model = MLP_multilayer(
            n_inputs= model_config["no_of_inputs"],
            layers_defs= model_config["parameters"],
            n_classes= model_config["no_of_classes"],
            use_F0_too_low_class = labels_transformer_options["use_F0_too_low_class"]) # creation of model topology
    else:
        m_type = model_config["type"]
        raise Exception(f"Unexpected network model type: {m_type}")

    are_classes_labels_saved = False
    for input_wav_files_entry in input_wav_files:
        input_root_folder = input_wav_files_entry[0]
        input_subfolder = input_wav_files_entry[1]
        input_filenames = input_wav_files_entry[2]

        logging.info(f"input_root_folder: {input_root_folder.as_posix()}")
        logging.info(f"  subfolder: {input_subfolder.as_posix()}")

        for wav_filename in input_filenames:
            logging.info(f">> wav_filename: {wav_filename}")

            wav_file_path = PurePath(input_root_folder, input_subfolder, wav_filename)

            # TODO either under wav file path or path given as a third argument ???
            # epoch_infer_save_path = PurePath(output_data_path, PurePath(model_folder).name, epoch_str)
            epoch_infer_save_path = PurePath(output_data_path, input_subfolder, PurePath(wav_filename).stem).with_suffix("."+output_file_format)

            DATA_INPUT_PATH = { "PATH": wav_file_path.parent, 
                    "F0_ref_filename": None,
                    "WAV_filename": wav_file_path.name,
                    "FB_data_filename": None,
            }
            inference_dataset = create_data_from_file(DATA_INPUT_PATH, dataset_options) # create dataset based on given data path

            # labels conversion
            inference_dataset.convert_labels(labels_transformer_options)
            # classes_labels = inference_dataset.classes_labels
            if are_classes_labels_saved == False:
                # save just once since the same model is used for all files
                inference_dataset.save_classes_labels(Path(output_data_path))
                are_classes_labels_saved = True

            train_accuracy = checkpoint[ModelBinaryAttributes.ACCURACY]
            accuracy_table = checkpoint[ModelBinaryAttributes.ACCURACY_TABLE]
            model_data_2_save = {"model_training_accuracy": {"train_accuracy": train_accuracy, "accuracy_table": accuracy_table, "batch_size": checkpoint_trainer_config["batch_size"]}}
            logging.info(f">> train_accuracy={train_accuracy}")
            try:
                infer_results = infer_model(network_model, checkpoint, inference_dataset, checkpoint_trainer_config["batch_size"], model_data_2_save, 
                                            epoch_infer_save_path = epoch_infer_save_path, save_filename_prefix = "",
                                            output_file_format = output_file_format) # running evaluation engine
                # classes_labels = ???

            except Exception as e:
                logging.info(" >>> infer_model failed !!!")
                logging.info(traceback.format_exc())
                # break
    
    #
    logging.info("END")
    if do_clear_log == True:
        log_cfg.reinit_additional_output_file() # close it

def main_function():
    if len(sys.argv) <= 4:
        print(f'{PurePath(sys.argv[0]).name} requires: (1) wav file name, (2) trained model filename, (3) output directory, and [optionaly] output file format ("npz" (default) or "mat")')
        exit()
    else:
        output_data_path = sys.argv[3]
        print(f"output_data_path: {output_data_path}")
        global_inference_output_data_path = Path(output_data_path).resolve()
        if not global_inference_output_data_path.exists():
            global_inference_output_data_path.mkdir(parents=True, exist_ok=True)
            if not global_inference_output_data_path.exists():
                raise Exception(f"output path: {global_inference_output_data_path} does not exist")
            
        log_cfg = LogConfig()
        log_cfg.init_logging("IFE_infer", output_data_path)

        logging.info(f'{PurePath(sys.argv[0]).name} started with arguments: {sys.argv[1:]}')
        logging.info(f"output_data_path: {output_data_path}")

        if len(sys.argv) == 5:
            output_file_format = sys.argv[4]
        else:
            output_file_format = "npz"

        wav_filename = sys.argv[1]
        validation_data_source = 'file'
        logging.info(f"wav_filename: {wav_filename}")
        wav_file_path = Path(wav_filename)

        # TODO test this & modify the code so all returend files can be processed
        input_wav_files = get_input_files(wav_file_path)
        if input_wav_files is None:
            raise Exception(f"wav file or folder with wav files: {wav_filename} does not exist")

        # raise Exception(f"Unsupported processing of list of wav files")

        model_filename = sys.argv[2]
        logging.info(f"model_filename: {model_filename}")
        model_path = Path(model_filename)
        if not model_path.exists():
            raise Exception(f"model file: {model_filename} does not exist")

        do_overwrite = True
        logging.info(f"do_overwrite: {do_overwrite}")

        # logging.info(f"log_cfg.additional_logging_file_handler({wav_file_path.parent},{model_path.name})")
        # log_cfg.reinit_additional_output_file(wav_file_path.parent,model_path.name)

        # SNR_dB_flag_idx = -1
        # SNR_dB_vector = [None]

        # TODO IFE_infere => signal that logging is initialized od initialize it for stdout
        IFE_infer(input_wav_files = input_wav_files, model_path = model_path, output_data_path = global_inference_output_data_path, log_cfg = log_cfg, output_file_format = output_file_format)
    
        log_cfg.reinit_additional_output_file() # close it

if __name__ == "__main__":
    main_function()
