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

def get_model_files(model_path, filename_mask):
    files = [file.name for file in Path(model_path).glob(filename_mask)]
    return files

log_cfg = LogConfig()
log_cfg.init_logging("infer_model")

logging.info(f'{PurePath(sys.argv[0]).name} started with arguments: {sys.argv[1:]}')

if len(sys.argv) <= 1:
    logging.error(f'{PurePath(sys.argv[0]).name} requires config json file(s) as argument(s)')
    exit()
else:
    for argument in sys.argv[1:]:
        logging.info(f"Loading config from: {argument}")
        # TODO check if file has been successfuly open and read
        with open(argument, 'rt') as f:
            inferer_configuration_data = json.load(f)
            logging.info(inferer_configuration_data)

        # process loaded config data
        validation_data_configs = inferer_configuration_data["validation_data"]

        for validation_data_options in validation_data_configs:
            logging.info("Validation data option:")
            logging.info(validation_data_options)

            if not "skip" in validation_data_options:
                validation_data_options["skip"] = False
            if validation_data_options["skip"] == True:
                logging.info("Skipping validation_data_options entry")

            else:

                for dataset_options in inferer_configuration_data["dataset_options"]:
                    # loaded_options basic info
                    logging.info("Dataset options:")
                    logging.info(dataset_options)

                    for inferer_options in inferer_configuration_data["inferer_options"]:
                        logging.info("Inferer options:")
                        logging.info(inferer_options)

                        if not "skip" in inferer_options:
                            inferer_options["skip"] = False
                        if inferer_options["skip"] == True:
                            logging.info("Skipping inferer_options entry")

                        else:

                            for inferer_config in inferer_options["inferer_configs"]:
                                model_files_cfg = inferer_config["model_files_cfg"]
                                # determine list of  model data folders
                                model_root_path = inferer_options["root_path"] + model_files_cfg["subpath"]
                                model_folders = get_model_folders(model_root_path, model_files_cfg["folder_prefix"], model_files_cfg["folder_name"])

                                if "overwrite" in inferer_config:
                                    do_overwrite = inferer_config["overwrite"] 
                                else:
                                    do_overwrite = False
                                    logging.info("overwrite not defined in inferer_config: defaulting to False")

                                for model_folder in model_folders:
                                    # TODO determine list of model data files for each epoch

                                    global_inference_output_data_path_str = validation_data_options["global_inference_output_data_path"]

                                    # preload data without labels conversion
                                    validation_data_source = validation_data_options["mode"]
                                        
                                    if validation_data_source == "synthesis":
                                        SNR_dB_flag_idx =  global_inference_output_data_path_str.find(r"{SNR_dB}")
                                    else:
                                        SNR_dB_flag_idx = -1

                                    if "SNR_dB" in validation_data_options:
                                        SNR_dB_range = validation_data_options["SNR_dB"]
                                        if SNR_dB_flag_idx > -1:
                                            SNR_dB_vector = np.arange(SNR_dB_range[0], SNR_dB_range[1]+SNR_dB_range[2]/10, SNR_dB_range[2])
                                        else:
                                            SNR_dB_vector = SNR_dB_range[0]
                                    else:
                                        SNR_dB_vector = [None]


                                    for SNR_dB_value in SNR_dB_vector:
                                        logging.info(f"SNR_dB_value = {SNR_dB_value}dB")

                                        if SNR_dB_flag_idx > -1:
                                            global_inference_output_data_path_str_with_SNR = global_inference_output_data_path_str[:SNR_dB_flag_idx+1] # leave '{'
                                            global_inference_output_data_path_str_with_SNR += f"{SNR_dB_value:.3f}"
                                            global_inference_output_data_path_str_with_SNR += global_inference_output_data_path_str[SNR_dB_flag_idx+8-1:] # leave '}'

                                            global_inference_output_data_path = PurePath("output_data/inference/", global_inference_output_data_path_str_with_SNR)

                                        else:                                      
                                            global_inference_output_data_path = PurePath("output_data/inference/", validation_data_options["global_inference_output_data_path"])
                                        
                                        logging.info(f"log_cfg.additional_logging_file_handler({global_inference_output_data_path},{Path(model_folder).name})")
                                        log_cfg.reinit_additional_output_file(global_inference_output_data_path,Path(model_folder).name)

                                        if len(model_files_cfg["epochs_indexes"]) > 0:
                                            filename_mask = model_files_cfg["filename_prefix"] + "*"
                                            model_files = get_model_files(model_folder, filename_mask)

                                            indexes = range(len(model_files))
                                            indexes = eval("indexes[" + model_files_cfg["epochs_indexes"] +"]")
                                            if isinstance(indexes, int):
                                                indexes = [indexes]

                                            model_files_tmp = []
                                            for idx in indexes:
                                                filename_tmp = filename_mask[:-1] + str(idx)
                                                if filename_tmp in model_files:
                                                    model_files_tmp.append(filename_tmp)
                                                else:
                                                    logging.error(f"could not find {filename_tmp} file")
                                            model_files = model_files_tmp
                                        else:
                                            filename_mask = model_files_cfg["filename_prefix"]
                                            model_files = get_model_files(model_folder, filename_mask)

                                        if len(model_files) == 0:
                                            logging.error(f"Could not find trained models in the given folder: {model_folder}\n")
                                            
                                        else:
                                            # TODO check if it works for direct filename in model_files_cfg["filename_prefix"]
                                            model_path = model_folder + "/" + model_files[0]

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

                                            if "model_config" in inferer_config:
                                                model_config_ref = inferer_config["model_config"]

                                                if model_config is None:
                                                    model_config = model_config_ref 
                                                else:
                                                    # compare with config from loaded model
                                                    if Model_auxdata.compare_model_configs(model_config, model_config_ref) == False:
                                                        logging.error("Model from inferer configuration differs from trained model: using the trained model")
                                            if model_config is None:
                                                raise Exception("model_config could not be determined")

                                            # TODO compare dataset options
                                            dataset_options_tmp = checkpoint_dataset_options
                                            dataset_options_tmp["shuffle_data"] = dataset_options["shuffle_data"]
                                            dataset_options_tmp["maintain data continuity"] = dataset_options["maintain data continuity"]
                                            dataset_options = dataset_options_tmp

                                            if ModelBinaryAttributes.TRAINING_DATASET_OPTIONS in checkpoint:
                                                checkpoint_training_dataset_options = checkpoint[ModelBinaryAttributes.TRAINING_DATASET_OPTIONS]
                                            else:
                                                checkpoint_training_dataset_options = None

                                            if "labels transformer options" in inferer_config:
                                                labels_transformer_options_ref = inferer_config["labels transformer options"]
                                                labels_transformer_options_ref["no_of_classes"] = model_config["no_of_classes"]
                                                
                                                if labels_transformer_options is None:
                                                    labels_transformer_options = labels_transformer_options_ref 
                                                else:
                                                    # compare with config from loaded model
                                                    if LabelsTransformer.compare_labels_configs(labels_transformer_options, labels_transformer_options_ref) == False:
                                                        logging.error("label transformer configuration from inferer configuration differs from trained model configuration: using the trainded model configuration")
                                            if labels_transformer_options is None:
                                                raise Exception("labels_transformer_options could not be determined")

                                            validation_dataset = None


                                            # for each epoch
                                            for model_file_idx in range(len(model_files)):
                                                if model_file_idx > 0: # first model is already loaded
                                                    model_path = model_folder + "/" + model_files[model_file_idx]
                                                    # Load model state and configuration
                                                    checkpoint = torch.load(PurePath(model_path).as_posix())

                                                    if ModelBinaryAttributes.DATASET_OPTIONS_OLD in checkpoint:
                                                        checkpoint_dataset_options = checkpoint[ModelBinaryAttributes.DATASET_OPTIONS_OLD]
                                                    if ModelBinaryAttributes.DATASET_OPTIONS in checkpoint:
                                                        checkpoint_dataset_options = checkpoint[ModelBinaryAttributes.DATASET_OPTIONS]
                                                    if ModelBinaryAttributes.LABEL_TRANSFORMER_OPTIONS in checkpoint:
                                                        checkpoint_label_transformer_options = checkpoint[ModelBinaryAttributes.LABEL_TRANSFORMER_OPTIONS]
                                                        #labels_transformer_options = checkpoint_label_transformer_options
                                                    if ModelBinaryAttributes.TRAINER_CONFIG in checkpoint:
                                                        checkpoint_trainer_config = checkpoint[ModelBinaryAttributes.TRAINER_CONFIG]
                                                        # use model_config from checkpoint
                                                        #model_config = checkpoint_trainer_config["model_config"]
                                                    # TODO verify checkpoint configs

                                                # ??????????????????????????????????????????
                                                # TODO extract parameters from checkpoint
                                                model_epoch = checkpoint[ModelBinaryAttributes.EPOCH]
                                                epoch_str = "epoch_" + str(model_epoch)
                                                logging.info(f"Epoch: {epoch_str}")

                                                # epoch_infer_save_path = model_path + "_inference"
                                                epoch_infer_save_path = PurePath(global_inference_output_data_path, PurePath(model_folder).name, epoch_str)

                                                model_params_filename = epoch_infer_save_path.parent / "model_params.json"
                                                if not Path(model_params_filename).is_file():
                                                    # save model parameter in json file
                                                    Path(model_params_filename).parent.mkdir(parents=True, exist_ok=True)
                                                    model_params = {
                                                            "model_path": model_path,
                                                            "model_config": model_config,
                                                            "labels_transformer_options": labels_transformer_options,
                                                            "checkpoint_trainer_config": checkpoint_trainer_config,
                                                            "validation_data_options": validation_data_options,
                                                            "dataset_options": dataset_options,
                                                            "training_dataset_options": checkpoint_training_dataset_options # from first epoch
                                                        }
                                                    logging.info(f"Saving parameters to: {model_params_filename}")
                                                    with open(model_params_filename,'w+') as f:
                                                        json.dump(model_params, f, indent=4)            

                                                skip_current_epoch = False
                                                if do_overwrite == False:
                                                    # check if results files alredy exist
                                                    results_filename = epoch_infer_save_path.with_name("infer_data_" + epoch_infer_save_path.name)
                                                    if Path(results_filename).with_suffix(".npz").is_file():
                                                        skip_current_epoch = True
                                                    
                                                if skip_current_epoch == True:
                                                    logging.info("Results file already exist skipping...")
                                                else:
                                                    if validation_dataset is None:
                                                        if validation_data_source == "file":
                                                            validation_data_path = validation_data_options["root_path"] + validation_data_options["subpath"]
                                                            if "WAV_filename" not in validation_data_options:
                                                                validation_data_options["WAV_filename"] = None
                                                            if "FB_data_filename" not in validation_data_options:
                                                                validation_data_options["FB_data_filename"] = None
                                                            DATA_INPUT_PATH = { "PATH": validation_data_path, # path to training data folder
                                                                    "F0_ref_filename": validation_data_options["F0_ref_filename"],
                                                                    "WAV_filename": validation_data_options["WAV_filename"],
                                                                    "FB_data_filename": validation_data_options["FB_data_filename"],
                                                            }

                                                            dataset_options["shuffle_data"] = False # Force shuffle off
                                                            validation_dataset = create_data_from_file(DATA_INPUT_PATH, dataset_options) # create dataset based on given data path
                                                        elif validation_data_source == "synthesis":
                                                            # SNR_dB_range = validation_data_options["SNR_dB"]
                                                            no_of_segments = validation_data_options["no_of_segments"]
                                                            Fs = validation_data_options["Fs"]
                                                            segment_length = validation_data_options["segment_length"]
                                                            silence_length = validation_data_options["silence_length"]
                                                            F0_min = validation_data_options["F0_min"]
                                                            F0_max = validation_data_options["F0_max"]

                                                            if "save_wav" in validation_data_options:
                                                                save_validation_audio_to_wav = validation_data_options["save_wav"]
                                                            else:
                                                                save_validation_audio_to_wav = False

                                                            # rng_state_filename = None
                                                            if not "rng_state_filename" in validation_data_options:
                                                                validation_data_options["rng_state_filename"] = "rng_state.entropy"
                                                            rng_state_filename = global_inference_output_data_path / validation_data_options["rng_state_filename"]

                                                            dataset_options["shuffle_data"] = False # Force shuffle off
                                                            rng_state_idx = 0

                                                            if save_validation_audio_to_wav == True:
                                                                validation_dataset = create_data_by_synthesis(SNR_dB_value, no_of_segments, segment_length, silence_length, F0_min, F0_max, Fs, rng_state_filename, rng_state_idx, 
                                                                                                            SNR_index = -1, save_validation_audio_to_wav_path = global_inference_output_data_path,
                                                                                                            dataset_options=dataset_options) # create dataset 
                                                            else:
                                                                validation_dataset = create_data_by_synthesis(SNR_dB_value, no_of_segments, segment_length, silence_length, F0_min, F0_max, Fs, rng_state_filename, rng_state_idx, 
                                                                                                            SNR_index = -1, save_validation_audio_to_wav_path = None,
                                                                                                            dataset_options=dataset_options) # create dataset 
                                                        else:
                                                            logging.error(f"Wrong validation_data_source: {validation_data_source}")

                                                        # Saving reference F0_ref
                                                        validation_dataset.save_F0_ref(global_inference_output_data_path, validation_data_options)

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

                                                    # labels conversion
                                                    validation_dataset.convert_labels(labels_transformer_options)
                                                    validation_dataset.save_classes_labels(Path(epoch_infer_save_path.parent))
                                                    classes_labels = validation_dataset.classes_labels

                                                    train_accuracy = checkpoint[ModelBinaryAttributes.ACCURACY]
                                                    accuracy_table = checkpoint[ModelBinaryAttributes.ACCURACY_TABLE]
                                                    model_data_2_save = {"model_training_accuracy": {"train_accuracy": train_accuracy, "accuracy_table": accuracy_table, "batch_size": model_files_cfg["batch_size"]}}
                                                    logging.info(f">> train_accuracy={train_accuracy}")
                                                    try:
                                                        infer_results = infer_model(network_model, checkpoint, validation_dataset, model_files_cfg["batch_size"], model_data_2_save, epoch_infer_save_path = epoch_infer_save_path) # running evaluation engine
                                                        # classes_labels = ???

                                                    except Exception as e:
                                                        logging.info(" >>> infer_model failed !!!")
                                                        logging.info(traceback.format_exc())
                                                        break

                                                    # # ploting 
                                                    # if epoch_dp.get_classes_labels() == None:
                                                    #     epoch_dp.set_classes_labels(classes_labels)
                                                    # 
                                                    # # if (model_epoch % draw_epoch_step == 0) or (model_epoch == no_of_epochs-1):
                                                    # if (model_epoch % draw_epoch_step == 0) or (model_file_idx == len(model_files)-1):
                                                    #     epoch_dp.append_epoch_data(infer_results, train_accuracy, infer_results["save_path"], model_epoch, do_draw_epochs)
                                                    # else:
                                                    #     epoch_dp.append_epoch_data(infer_results, train_accuracy, infer_results["save_path"], model_epoch, False)

                                            #
                                            logging.info("END")

                                log_cfg.reinit_additional_output_file() # close it
