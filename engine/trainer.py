from tqdm import tqdm
from torch import torch, optim, nn
from datetime import datetime
from shutil import copyfile
from pathlib import PurePath, Path
import logging
import os

from torch.utils.data import DataLoader

from utils.cuda_utils import get_cuda_device_with_info
from utils.constants import ModelBinaryAttributes
from data_processor.creators import create_data_from_file, create_data_by_synthesis
from data_processor.labels_transformer import LabelsTransformer
from networks.MLP_2_layers import Model_auxdata, MLP_2_layers, MLP_multilayer, MLP_3_layers


def get_model_folder(model_path, folder_prefix, folder_name):
    subfolder = model_path + folder_prefix + folder_name
    return subfolder

def get_model_files(model_path, filename_mask):
    files = [file.name for file in Path(model_path).glob(filename_mask)]
    return files

def load_and_check_model(model_path, trainer_options):
    checkpoint = torch.load(PurePath(model_path).as_posix())

    # extract a configuration subset used for particular model
    
    # if ModelBinaryAttributes.DATASET_OPTIONS in checkpoint:
    #     checkpoint_dataset_options = checkpoint[ModelBinaryAttributes.DATASET_OPTIONS]
    if ModelBinaryAttributes.LABEL_TRANSFORMER_OPTIONS in checkpoint:
        checkpoint_label_transformer_options = checkpoint[ModelBinaryAttributes.LABEL_TRANSFORMER_OPTIONS]
    else:
        logging.error("no label_transformer_options in checkpoint")
        return None
    if ModelBinaryAttributes.TRAINER_CONFIG in checkpoint:
        checkpoint_trainer_config = checkpoint[ModelBinaryAttributes.TRAINER_CONFIG]
    else:
        logging.error("no trainer_config in checkpoint")
        return None
    if ModelBinaryAttributes.MODEL_CONFIG in checkpoint:
        checkpoint_model_config = checkpoint[ModelBinaryAttributes.MODEL_CONFIG]
    else:
        logging.error("no model_config in checkpoint")
        return None

    if "model_config" in trainer_options:
        model_config_ref = trainer_options["model_config"]

        # compare with config from loaded model
        if Model_auxdata.compare_model_configs(checkpoint_model_config, model_config_ref) == False:
            logging.error("Model from trainer configuration differs from previously trained model")
            return None
    else:
        return None

    if "labels transformer options" in trainer_options:
        labels_transformer_options_ref = trainer_options["labels transformer options"]
        labels_transformer_options_ref["no_of_classes"] = model_config_ref["no_of_classes"]

        checkpoint_labels_transformer_options_ref = checkpoint_label_transformer_options
        checkpoint_labels_transformer_options_ref["no_of_classes"] = checkpoint_model_config["no_of_classes"]
        
        # compare with config from loaded model
        if LabelsTransformer.compare_labels_configs(checkpoint_labels_transformer_options_ref, labels_transformer_options_ref) == False:
            logging.error("label transformer configuration from inferer configuration differs from trained model configuration: using the trainded model configuration")
            return None

    return checkpoint

def train_model(train_dataset, training_data_options, dataset_options, trainer_options, log_cfg = None):
    """Trains model with given data"""

    reload_step = training_data_options["reload"]
    training_data_source = training_data_options["mode"]

# train_dataset, DATA_INPUT_PATH
    # 0. train_dataset = None 
    # 1. initialize network
    # 1b. if train_dataset != None: Convert to labels
    # 2. if train_dataset = None: Load/synthesize data
    # 3. Convert to labels
    # 4. if reload_step == 0
    #      train all requested epochs
    #    else 
    #      train until (epoch_ind mod reload_step) or all requested epochs
    # 5. if not last epoch:   
    #    if (epoch_ind mod reload_step) == 0:
    #      train_dataset = None
    #      go to 2
    #    else: go to 4
    # 6. Change label convert method
    #  if reload_step > 0: train_dataset = None
    #  go to 2
    # 7. change network model
    #  if reload_step > 0: train_dataset = None
    #  go to 2


    labels_transformer_options = trainer_options["labels transformer options"]
    model_config = trainer_options["model_config"]
    trainer_config = trainer_options["trainer_config"]

    labels_transformer_options["no_of_classes"] = model_config["no_of_classes"]

    gpu_device = get_cuda_device_with_info() # print details about available devices and return first cuda gpu

    # #################################### #
    # 1. initialize network
    # #################################### #
    #model_save_path = Path(f"output_data\models\TRAINING_{datetime.now().strftime('%Y%m%d-%H%M%S')}{model.tag_str}")
    model_tag_str = Model_auxdata.get_tag_from_options(training_data_options, dataset_options, labels_transformer_options, trainer_config, model_config)
    if trainer_config["folder_name"] is None:
        folder_name = f"{model_tag_str}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    elif len(trainer_config["folder_name"]) == 0:
        folder_name = f"{model_tag_str}"
    else:
        folder_name = trainer_config["folder_name"]
    root_path = "output_data/models/" + trainer_config["subpath"]
    model_save_path = Path(get_model_folder(root_path, trainer_config["folder_prefix"], folder_name))

    logging.info(model_save_path)
    if log_cfg is not None:
        model_save_path.mkdir(parents=True, exist_ok=True)
        log_cfg.reinit_additional_output_file(model_save_path, "training_log")

    # get list of already available output files
    if "filename_prefix" not in trainer_config:
        trainer_config["filename_prefix"] = "epoch_"
    filename_mask = trainer_config["filename_prefix"] + "*"
    model_files = get_model_files(model_save_path, filename_mask)

    # place files in order
    indexes = range(len(model_files))
    model_files_tmp = []
    for idx in indexes:
        filename_tmp = filename_mask[:-1] + str(idx)
        if filename_tmp in model_files:
            model_files_tmp.append(filename_tmp)
        else:
            logging.error(f"could not find {filename_tmp} file")
    model_files = model_files_tmp

    # determine first epoch index to use / determine last available epoch, if any (starting epoch index)
    logging.info(f" ####################################################### ")
    if trainer_config["overwrite"] == False:
        epoch_start_idx = len(model_files)
        logging.info(f"overwrite: False, epoch_start_idx={epoch_start_idx}")
    else:
        # on first epoch write delete previous files
        # TODO determine list of files to delete
        epoch_start_idx = 0
        logging.info(f"overwrite: True, epoch_start_idx={epoch_start_idx}")

    max_epochs = trainer_config["no_of_epochs"]
    if epoch_start_idx >= max_epochs:
        logging.info(f"epoch_start_idx({epoch_start_idx}) >= max_epochs({max_epochs}): skipping training")
        return
    logging.info(f"Training {max_epochs} epochs")

    # select and create model
    # TODO add parameters and other model types
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

    # load staring epoch final state
    checkpoint = None
    if epoch_start_idx > 0:
        logging.info(f"Loading network model from previous epoch")
        filename_prefix = trainer_config["filename_prefix"]
        model_path = PurePath(model_save_path / f"{filename_prefix}{epoch_start_idx-1}").as_posix()
        logging.info(f"model_path: {model_path}")
        checkpoint = load_and_check_model(model_path, trainer_options)

    if checkpoint is not None:
        logging.info(f"initializing network model with state from previous epoch")
        network_model.load_state_dict(checkpoint[ModelBinaryAttributes.MODEL_STATE_DICT])
        network_model.to(gpu_device) # move model to GPU
        optimizer = optim.Adam(network_model.parameters(), lr=trainer_config["learning_rate"])
        optimizer.load_state_dict(checkpoint[ModelBinaryAttributes.OPTIMIZER_STATE_DICT])
    else:
        epoch_start_idx = 0
        logging.info(f"initializing new network model")
        network_model.to(gpu_device) # move model to GPU
        optimizer = optim.Adam(network_model.parameters(), lr=trainer_config["learning_rate"]) # create optimizer for backpropagation
    loss_function = nn.CrossEntropyLoss() # choose loss function for learning rating
    logging.info(f"Model Summary:\n{network_model}\n")

    # #################################### #
    # 1b. if train_dataset != None: Convert to labels
    # #################################### #
    if train_dataset is not None:
        train_dataset.convert_labels(labels_transformer_options)

    current_epoch_idx = epoch_start_idx
    while current_epoch_idx < max_epochs:
        # #################################### #
        # 2. if train_dataset = None: Load/synthesize data
        # #################################### #
        if train_dataset is None:
            # preload data without labels conversion
            if training_data_source == "file":
                training_data_path = training_data_options["root_path"] + training_data_options["subpath"]
                if "WAV_filename" not in training_data_options:
                    training_data_options["WAV_filename"] = None
                if "FB_data_filename" not in training_data_options:
                    training_data_options["FB_data_filename"] = None
                DATA_INPUT_PATH = { "PATH": training_data_path, # path to training data folder
                        "F0_ref_filename": training_data_options["F0_ref_filename"],
                        "WAV_filename": training_data_options["WAV_filename"],
                        "FB_data_filename": training_data_options["FB_data_filename"],
                }
                train_dataset = create_data_from_file(DATA_INPUT_PATH, dataset_options) # create dataset based on given data path
            elif training_data_source == "synthesis":
                SNR_dB_range = training_data_options["SNR_dB"]
                no_of_segments = training_data_options["no_of_segments"]
                Fs = training_data_options["Fs"]
                segment_length = training_data_options["segment_length"]
                silence_length = training_data_options["silence_length"]
                F0_min = training_data_options["F0_min"]
                F0_max = training_data_options["F0_max"]

                rng_state_filename = None
                #if current_epoch_idx == 0:
                if not "rng_state_filename" in training_data_options:
                    training_data_options["rng_state_filename"] = None
                if training_data_options["rng_state_filename"] is not None:
                    rng_state_filename = model_save_path / training_data_options["rng_state_filename"]

                if reload_step == 0:
                    rng_epoch_idx = 0
                elif reload_step == 1:
                    rng_epoch_idx = current_epoch_idx
                else:
                    rng_epoch_idx = current_epoch_idx - (current_epoch_idx % reload_step)

                train_dataset = create_data_by_synthesis(SNR_dB_range, no_of_segments, segment_length, silence_length, F0_min, F0_max, Fs, rng_state_filename, rng_epoch_idx, 
                                                         SNR_index = current_epoch_idx, dataset_options=dataset_options) # create dataset 
            else:
                logging.error(f"Wrong training_data_source: {training_data_source}")

            # #################################### #
            # 3. Convert to labels
            # #################################### #
            train_dataset.convert_labels(labels_transformer_options)

        # #################################### #
        network_model.store_options(training_data_options = training_data_options, 
            dataset_options = train_dataset.dataset_options,
            labels_transformer_options = train_dataset.labels_transformer_options,
            trainer_config = trainer_config,
            model_config = model_config)
        # #################################### #

        train_data_loader = DataLoader(train_dataset, shuffle=trainer_config["shuffle_data"], batch_size=trainer_config["batch_size"]) # just initialization !!! TODO check this
        logging.info(f"Done!")

        # #################################### #
        # 4. if reload_step == 0
        #      train all requested epochs
        #    else 
        #      train until (epoch_ind mod reload_step) or all requested epochs
        # #################################### #
        # run training step
        if reload_step == 0:
            train(model=network_model,
                data_loader=train_data_loader,
                SNR_dB = train_dataset.SNR_dB,
                optimizer=optimizer,
                loss_function=loss_function,
                device=gpu_device,
                epoch_start_idx=epoch_start_idx,
                epochs=max_epochs,
                previous_model_files=model_files,
                batch_size=trainer_config["batch_size"],
                filename_prefix=trainer_config["filename_prefix"],
                save_path=model_save_path)
            current_epoch_idx = max_epochs
        else:
            # calculate number of epoch to run until reload
            current_max_epochs = current_epoch_idx + (reload_step - (current_epoch_idx % reload_step))
            train(model=network_model,
                data_loader=train_data_loader,
                SNR_dB = train_dataset.SNR_dB,
                optimizer=optimizer,
                loss_function=loss_function,
                device=gpu_device,
                epoch_start_idx=current_epoch_idx,
                epochs=current_max_epochs,
                previous_model_files=model_files,
                batch_size=trainer_config["batch_size"],
                filename_prefix=trainer_config["filename_prefix"],
                save_path=model_save_path)
            current_epoch_idx = current_max_epochs
            train_dataset = None # force data reload/resynthesis
    
    if log_cfg is not None:
        log_cfg.reinit_additional_output_file() # reset additional log

    if reload_step > 0:
        train_dataset = None # force data reload/resynthesis on next setup
    return train_dataset

def train(model, data_loader, SNR_dB, optimizer, loss_function, device, batch_size, epoch_start_idx, epochs, previous_model_files, filename_prefix, save_path):
    """Common function for training models"""
    running_loss = 0
    epochs_results = {}

    no_of_batches = data_loader.__len__()

    accuracy_step = int(no_of_batches / 500) # update five times per one percent
    for epoch in range(epoch_start_idx,epochs):
        logging.info(f"Training epoch: {epoch}")
        epoch_loss = 0
        total = 0
        correct = 0
        accuracy_table = []
        with tqdm(data_loader, unit=" batches", mininterval=0.2, dynamic_ncols=True) as tepoch:
            for batch_id, (data, labels, ref_values) in enumerate(tepoch):
                data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.int64).to(device)
                optimizer.zero_grad()
                prediction = model(data)
                # TODO use ref_values in loss_funtion
                loss = loss_function(prediction, labels)
                loss.backward()
                optimizer.step()

                _, predicted = prediction.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                running_loss += loss.item()
                epoch_loss += loss.item()

                if batch_id % batch_size == 0: #update once per batch
                    accuracy_table.append(100.*correct/total)

                # TODO decrease update rate
                if batch_id % accuracy_step == 0: #update once per accuracy_step
                    tmp_accu=100.*correct/total
                    tepoch.set_postfix(loss=running_loss/batch_size, accu=tmp_accu)
                    running_loss = 0

        train_loss=epoch_loss/(len(data_loader) * batch_size)
        epoch_loss = 0
        accu=100.*correct/total

        logging.info('EPOCH %i Train Loss: %.3f | Accuracy: %.3f'%(epoch, train_loss, accu))

        epochs_results[str(epoch)] = accu

        # Prepare output folder
        save_path.mkdir(parents=True, exist_ok=True)

        # save model
        # torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': train_loss,
        #             "accuracy": accu
        #             }, PurePath(model_save_path / f"epoch_{epoch}").as_posix())
        if (epoch_start_idx == 0) and (epoch == 0):
            # delete all previous files
            for filename in previous_model_files:
                filename_withpath = PurePath(save_path / filename).as_posix()
                os.remove(filename_withpath)

        torch.save({
                    ModelBinaryAttributes.VERSION: 0x0004,
                    ModelBinaryAttributes.EPOCH: epoch,
                    ModelBinaryAttributes.MODEL_STATE_DICT: model.state_dict(),
                    ModelBinaryAttributes.OPTIMIZER_STATE_DICT: optimizer.state_dict(),
                    ModelBinaryAttributes.LOSS: train_loss,
                    ModelBinaryAttributes.ACCURACY: accu,
                    ModelBinaryAttributes.ACCURACY_TABLE: accuracy_table,
                    # ===================================== #
                    ModelBinaryAttributes.TRAINING_DATASET_OPTIONS: model.training_data_options,
                    ModelBinaryAttributes.TRAINING_DATASET_SNR: SNR_dB,
                    ModelBinaryAttributes.DATASET_OPTIONS: model.dataset_options,
                    ModelBinaryAttributes.LABEL_TRANSFORMER_OPTIONS: model.labels_transformer_options,
                    ModelBinaryAttributes.TRAINER_CONFIG: model.trainer_config,
                    ModelBinaryAttributes.MODEL_CONFIG: model.model_config
                    }, PurePath(save_path / f"{filename_prefix}{epoch}").as_posix())

    max_accuracy = max(epochs_results.values())
    epoch_number = max(epochs_results, key=epochs_results.get)

    logging.info(f"Best result: {max_accuracy}%, for epoch number: {epoch_number}")
    # copyfile(save_path / f"epoch_{epoch_number}", save_path / f"BEST_EPOCH")
