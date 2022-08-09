from torch import torch, optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import PurePath, Path
import logging

from torch.utils.data import DataLoader
from utils.cuda_utils import get_cuda_device_with_info
from utils.constants import ModelBinaryAttributes 
from scipy.io import savemat

def save_epoch_infer_data(F0_est, F0_ref_labels, model_training_accuracy, training_SNR_dB, save_path, save_filename_prefix = "infer_data_", output_file_format = "npz"):
    # save ifcerence results of current epoch

    if len(save_path.as_posix()) > 0:
        filename = save_path.with_name(save_filename_prefix + save_path.name)
        filename = filename.with_suffix("."+output_file_format)
    else:
        raise Exception("save_epoch_infer_data: empty save_path")

    Path(filename.parent).mkdir(parents=True, exist_ok=True)
    
    if output_file_format == "npz":
        # np.savez(filename, F0_est=F0_est, F0_ref_labels=F0_ref_labels)
        np.savez_compressed(filename, F0_est=F0_est, F0_ref_labels=F0_ref_labels, 
                            model_training_accuracy=model_training_accuracy, training_SNR_dB=training_SNR_dB) # test
    else :
        if output_file_format == "mat":
            data_dict = dict()
            data_dict["F0_est"] = F0_est
            if F0_ref_labels is not None:
                data_dict["F0_ref_labels"] = F0_ref_labels
            else:
                data_dict["F0_ref_labels"] = []
            data_dict["model_training_accuracy"] = model_training_accuracy
            data_dict["training_SNR_dB"] = training_SNR_dB
            savemat(filename, data_dict, do_compression = True) 
        else:
            raise Exception("save_epoch_infer_data: unsupported file format")

def infer_model(model, checkpoint, validation_dataset, batch_size, model_data_2_save, epoch_infer_save_path=None, save_filename_prefix = "infer_data_", output_file_format = "npz"):
    """Evaluates given model with validation data"""
    gpu_device = get_cuda_device_with_info() # print details about available devices and return first cuda gpu


    infer_data_loader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size) # just initialization !!! TODO check this

    # # read data after model verification
    # dataset, data_loader = create_data(input_data_path) # create dataset and dataloader based on given data path

    model.load_state_dict(checkpoint[ModelBinaryAttributes.MODEL_STATE_DICT])

    model.to(gpu_device)

    # run evaluation step
    if hasattr(validation_dataset, 'F0_ref'):
        infer_results = infer(model=model,
                            data_loader=infer_data_loader,
                            dataset=validation_dataset,
                            device=gpu_device)
    else:
        infer_results = infer_raw(model=model,
                            data_loader=infer_data_loader,
                            dataset=validation_dataset,
                            device=gpu_device)

    if "F0_ref_labels" not in infer_results:
        infer_results["F0_ref_labels"] = None

    if epoch_infer_save_path is not None:
        #infer_results["F0_ref"]
        logging.info(f"save_epoch_infer_data: {epoch_infer_save_path}")
        if ModelBinaryAttributes.TRAINING_DATASET_SNR in checkpoint:
            save_epoch_infer_data(F0_est = infer_results["F0_est"], F0_ref_labels = infer_results["F0_ref_labels"], 
                                model_training_accuracy = model_data_2_save["model_training_accuracy"], 
                                training_SNR_dB = checkpoint[ModelBinaryAttributes.TRAINING_DATASET_SNR], 
                                save_path = epoch_infer_save_path, save_filename_prefix = save_filename_prefix,
                                output_file_format = output_file_format)
        else:
            logging.error(f"save_epoch_infer_data: training_SNR_dB is not defined")
            save_epoch_infer_data(F0_est = infer_results["F0_est"], F0_ref_labels = infer_results["F0_ref_labels"], 
                                model_training_accuracy = model_data_2_save["model_training_accuracy"], 
                                training_SNR_dB = None, 
                                save_path = epoch_infer_save_path, save_filename_prefix = save_filename_prefix,
                                output_file_format = output_file_format)

    infer_results["save_path"] = epoch_infer_save_path

    # return infer_results, checkpoint[ModelBinaryAttributes.ACCURACY], checkpoint[ModelBinaryAttributes.EPOCH], dataset.classes_labels
    return infer_results

def infer(model, data_loader, dataset, device):
    """Common function fo evaluating models"""
    correct = 0
    total = 0
    predictions = []
    decoded_predictions = []
    F0_ref_labels = []
    F0_ref_values = []
    model.eval() # enable eval behavior

    no_of_batches = data_loader.__len__()
    accuracy_step = int(no_of_batches / 100) # update once per one percent

    with torch.inference_mode():
         #with tqdm(data_loader, unit=" batches", mininterval=0.3) as tepoch:
         with tqdm(data_loader, unit=" batches", dynamic_ncols=True) as tepoch:
            for batch_id, (data, labels, ref_values) in enumerate(tepoch):
                #data, labels = data.to(device), labels.to(device)
                data = data.to(device)
                out_data = model(data)
                _, predicted = torch.max(out_data.data, 1)
                # total += predicted.size(0)

                current_F0_ref = ref_values.cpu().numpy()
                F0_ref_values.append(current_F0_ref)
                # valid_indexes = list(np.argwhere(current_F0_ref).tolist())
                valid_indexes = np.nonzero(current_F0_ref != 0)[0]
                total += len(valid_indexes)

                predictions.append(pd.DataFrame(predicted.cpu().numpy()))
                decoded_predictions.append(pd.DataFrame(dataset.encoder.inverse_transform(predicted.cpu().numpy())))

                if dataset.contains_labels:
                    #correct += (predicted == labels).sum().item()
                    np_labels = labels.cpu().numpy()
                    if len(valid_indexes) > 0:
                        valid_pred = np.transpose(predictions[-1].to_numpy())[0][valid_indexes]
                        valid_labels = np_labels[valid_indexes]
                        correct += (valid_pred == valid_labels).sum().item()

                    #F0_ref_labels.append(pd.DataFrame(labels.cpu().numpy()))
                    F0_ref_labels.append(dataset.encoder.inverse_transform(np_labels))
                else:
                    correct = np.nan
                    F0_ref_labels.append(None)


                if batch_id % accuracy_step == 0: #update once per percent
                    if total > 0:
                        tepoch.set_postfix(accu=100.*correct/total)

    if correct is not np.nan:
        logging.info('Accuracy of the network: %d %%' % (100 * correct / total))

    #F0_ref = get_reference_values(dataset)
    F0_ref = np.concatenate(F0_ref_values)
    if F0_ref_labels[0] is not None:
        F0_ref_labels = np.concatenate(F0_ref_labels)
    else:
        F0_ref_labels = None
    F0_est = get_estimated_values(dataset, decoded_predictions)

    return {"F0_ref": F0_ref, "F0_est": F0_est, "F0_ref_labels": F0_ref_labels}

def infer_raw(model, data_loader, dataset, device):
    """Common function fo evaluating models"""
    predictions = []
    decoded_predictions = []
    model.eval() # enable eval behavior

    no_of_batches = data_loader.__len__()

    with torch.inference_mode():
         #with tqdm(data_loader, unit=" batches", mininterval=0.3) as tepoch:
         with tqdm(data_loader, unit=" batches", dynamic_ncols=True) as tepoch:
            for batch_id, (data) in enumerate(tepoch):
                #data, labels = data.to(device), labels.to(device)
                data = data.to(device)
                out_data = model(data)
                _, predicted = torch.max(out_data.data, 1)
                # total += predicted.size(0)

                predictions.append(pd.DataFrame(predicted.cpu().numpy()))
                decoded_predictions.append(pd.DataFrame(dataset.encoder.inverse_transform(predicted.cpu().numpy())))

    F0_est = get_estimated_values(dataset, decoded_predictions)

    return {"F0_est": F0_est}

def get_reference_values(dataset):
    """Returns reference F0 values"""

    # #original_labels = dataset.encoder.inverse_transform(dataset.data_np[:,-1].astype(int))
    # original_labels = dataset.encoder.inverse_transform(dataset.F0_labels.astype(int))
    # F0_ref = np.transpose(original_labels)
    # return F0_ref
    return dataset.F0_ref

def get_estimated_values(dataset, decoded_predictions):
    """Returns estimated F0 values"""

    # decoded predictions (in Hz)
    decoded_output_data = pd.concat(decoded_predictions, ignore_index=True)
    F0_est = np.transpose(decoded_output_data[0].to_numpy())
    return F0_est

def get_save_path():
    # create directory for saving data
    save_path = Path(f"output_data/predictions/INFERENCE_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def save_data(dataset, predictions, decoded_predictions, save_path):
    """Saves csv with prediction results"""

    # save predictions
    output_data = pd.concat(predictions, ignore_index=True)
    output_data.to_csv(PurePath(save_path / "predictions.csv"), header=None, index=False)

    # save decoded predictions (in Hz)
    decoded_output_data = pd.concat(decoded_predictions, ignore_index=True)
    original_labels = dataset.encoder.inverse_transform(dataset.data_np[:,-1].astype(int))
    pd.DataFrame(original_labels).to_csv(PurePath(save_path / "decoded_original_labels.csv"), header=None, index=False)

def plotGraph(y_test,y_pred, regressorName, save_path):
    fig, axs = plt.subplots(3)
    axs[0].scatter(range(len(y_test)), y_test, color='blue', s=2)
    axs[0].scatter(range(len(y_pred)), y_pred, color='red', s=2)
    axs[0].set_title(regressorName)

    y_err = y_pred-y_test
    axs[1].scatter(range(len(y_test)), y_err, color='blue', s=2)
    axs[1].set_title('error')

    hist, bin_edges = np.histogram(y_err, bins=1000, density=True)
    bin_centers = (bin_edges[1:]+ bin_edges[:-1])/2
    axs[2].plot(bin_centers, hist)

    if save_path:
        fig.savefig(PurePath(save_path / "label_vs_prediction.png"), dpi=200)
    plt.show()
    return

