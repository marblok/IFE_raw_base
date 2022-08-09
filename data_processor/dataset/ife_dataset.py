from pathlib import PurePath, Path
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
import json
from sklearn.preprocessing import LabelEncoder

from data_processor.labels_transformer import LabelsTransformer
import data_processor.edges_calculators as ec

class IFEDataset_raw(Dataset):
    """Iterable dataset implementation specific for IFE dataset without F0_ref and labels"""
    def __init__(self, FB_data=None, FB_params = None, dataset_options=None, SNR = None):
    # def __init__(self, FB_data=None, FB_params = None, dataset_options=None, SNR = None):
        self.dataset_options = dataset_options
        self.labels_transformer_options = None
        self.SNR_dB = SNR

        logging.info(dataset_options)
        power_normalization = dataset_options["FB_data power normalization mode"]
        frequency_normalization = dataset_options["FB_data frequency normalization mode"]
        
        self.maintain_data_continuity = dataset_options["maintain data continuity"]
        self.do_shuffle = dataset_options["shuffle_data"]

        self.labels_transformer = None
        self.contains_labels = False

        self.FB_data = FB_data
        self.len = len(self.FB_data)
                
        self.FB_data_header = {}
        self.FB_data_header["Fc_max"] = FB_params.F_k_max
        self.FB_data_header["Fc_min"] = FB_params.F_0_min

        
        # DATA NORMALIZATION
        logging.info("FB_data normalization ...")
        # 1. normalize power entries (first element and then every second)
        #inst_powers = self.FB_data[:,0::2]
        max_P = np.amax(self.FB_data[:,0::2],1) # start:stop:step
        # https://www.delftstack.com/howto/numpy/python-numpy-divide-matrix-by-vector/
        logging.info(f"Instantaneous power normalization: {power_normalization}")
        if (power_normalization == "lin") or (power_normalization == "log"):
            self.FB_data[:,0::2] = self.FB_data[:,0::2] / max_P[:,None]
        else:
            logging.error(f"Skipping instantaneous power normalization")
        max_P = np.amax(self.FB_data[:,0::2],1) # start:stop:step

        if power_normalization == "log":
            logging.info("Applying log...")
            # TODO add threshold to load_option
            self.FB_data[:,0::2][self.FB_data[:,0::2] < 0.000001] = 0.000001
            self.FB_data[:,0::2] = np.log(self.FB_data[:,0::2])
            
        # 2. normalize frequencies
        logging.info(f"Instantaneous frequency normalization: {frequency_normalization}")
        if frequency_normalization == "lin_min":
            self.FB_data[:,1::2] = (self.FB_data[:,1::2]-self.FB_data_header["Fc_min"]) /(self.FB_data_header["Fc_max"]-self.FB_data_header["Fc_min"])
        elif (frequency_normalization == "lin_0") or (frequency_normalization == "log"):
            self.FB_data[:,1::2] = self.FB_data[:,1::2] / self.FB_data_header["Fc_max"]
        else:
            logging.error(f"Skipping instantaneous frequency normalization")

        if frequency_normalization == "log":
            logging.info("Applying log...")
            # TODO add threshold to load_option
            self.FB_data[:,1::2][self.FB_data[:,1::2] < 0.000001] = 0.000001
            self.FB_data[:,1::2] = np.log(self.FB_data[:,1::2])

        logging.info("FB_data normalization DONE")
        
    def convert_labels(self, labels_transformer_options):
        self.labels_transformer_options = labels_transformer_options

        self.classes_edges = None
        self.classes_labels = None
        
        use_F0_too_low_class = labels_transformer_options["use_F0_too_low_class"]
        if labels_transformer_options["mode"] is not None:
            F0_ref_min_value = labels_transformer_options["F0_ref_min"]
            F0_ref_max_value = labels_transformer_options["F0_ref_max"]
            no_of_classes = labels_transformer_options["no_of_classes"]

            # prepare for labels transforming
            if labels_transformer_options["mode"] == "lin":
                self.classes_labels = ec.get_lin_edges(F0_ref_min_value, F0_ref_max_value, no_of_classes)
                dF0_ref = self.classes_labels[1] - self.classes_labels[0]
                self.classes_edges = self.classes_labels - dF0_ref/2
                self.classes_edges = np.append(self.classes_edges, self.classes_labels[-1] + dF0_ref/2)
                linear_scale = True
            elif labels_transformer_options["mode"] == "log":
                self.classes_labels = ec.get_log_edges(F0_ref_min_value, F0_ref_max_value, no_of_classes)
                dF0_ref_factor = np.sqrt(self.classes_labels[1] / self.classes_labels[0])
                self.classes_edges = self.classes_labels / dF0_ref_factor
                self.classes_edges = np.append(self.classes_edges, self.classes_labels[-1] * dF0_ref_factor)
                linear_scale = False

            if use_F0_too_low_class:
                self.classes_labels = [np.float32(x) for x in np.insert(self.classes_labels, 0, 0.0)]
            else:
                self.classes_labels = [np.float32(x) for x in self.classes_labels]
            self.encoder = LabelEncoder()
            self.encoder.fit(self.classes_labels)

            if (not use_F0_too_low_class) and (not self.maintain_data_continuity):
                # add class at 0.0 just for discarding unfit data
                self.classes_labels = [np.float32(x) for x in np.insert(self.classes_labels, 0, 0.0)]
            
            self.labels_transformer = LabelsTransformer(labels_path = None, 
                min_value = F0_ref_min_value, max_value = F0_ref_max_value, 
                classes_labels = self.classes_labels, classes_edges = self.classes_edges, 
                classes_number = no_of_classes, use_F0_too_low_class=((not self.maintain_data_continuity) or use_F0_too_low_class), 
                linear_scale=linear_scale) # TODO linear_scale => add as parameter to configs

        self.contains_labels = False
        

    @staticmethod
    def load_classes_labels(load_path):
        # just returns classes labels loaded from file
        if len(load_path.as_posix()) > 0:
            filename = load_path / "infer_classes_labels"
        else:
            raise Exception("save_epoch_infer_data: empty load_path")

        json_filename = Path(filename).with_suffix(".json")
        if json_filename.is_file():
            with open(json_filename,'r+') as f:
                loaded_classes = json.load(f)
                return loaded_classes["classes_labels"]

        return None

    def save_classes_labels(self, save_path):
        if len(save_path.as_posix()) > 0:
            filename_labels = save_path / "infer_classes_labels"
            filename_F0_ref_indexes = save_path / "infer_F0_ref_indexes"
        else:
            raise Exception("save_epoch_infer_data: empty save_path")

        Path(save_path).mkdir(parents=True, exist_ok=True)

        # classes_filename = Path(filename).with_suffix(".npz")
        # if classes_filename.is_file():
        #     loaded_classes = np.load(classes_filename)
        #     res1 = np.array_equal(self.classes_labels, loaded_classes["classes_labels"])

        #     if res1:
        #         logging.info("skipping save_classes_labels: classes_labels file already saved")
        #     else:
        #         raise Exception(f"save_classes_labels: file {classes_filename} content differs from current classes labels")

        # np.savez_compressed(classes_filename, classes_labels=self.classes_labels) 

        if hasattr(self, 'indexes'):
            if Path(filename_F0_ref_indexes).with_suffix(".npz").is_file():
                loaded = np.load(Path(filename_F0_ref_indexes).with_suffix(".npz"))
                res2 = np.array_equal(self.indexes, loaded["F0_ref_indexes"])

                if res2:
                    logging.info("skipping: F0_ref_indexes file already saved")
                else:
                    full_filename = Path(filename_F0_ref_indexes).with_suffix(".npz")
                    raise Exception(f"save_classes_labels: file {full_filename} content differs from given F0_ref_indexes")

            else:
                # np.savez(filename, F0_ref=self.F0_ref, indexes=self.indexes)
                np.savez_compressed(filename_F0_ref_indexes, F0_ref_indexes=self.indexes) # test
        else:
            if Path(filename_F0_ref_indexes).with_suffix(".npz").is_file():
                raise Exception(f"save_classes_labels: no indexes but file {full_filename} with F0_ref_indexes ahs been detected")

        json_filename = Path(filename_labels).with_suffix(".json")
        if json_filename.is_file():
            with open(json_filename,'r+') as f:
                loaded_classes = json.load(f)
                res1 = np.array_equal(self.classes_labels, loaded_classes["classes_labels"])
                res2 = loaded_classes["maintain_data_continuity"] == self.maintain_data_continuity

                if res1 and res2:
                    logging.info("skipping save_classes_labels: classes_labels file already saved")
                    return
                else:
                    raise Exception(f"save_classes_labels: file {json_filename} content differs from current classes labels")

        with open(json_filename,'w+') as f:
            data = {"classes_labels": [float(x) for x in self.classes_labels],
                    "maintain_data_continuity": self.maintain_data_continuity}
            json.dump(data, f, indent=4)            

    def save_F0_ref(self, save_path, validation_data_options):
        logging.info("no F0_ref to save")
        


    def __getitem__(self, idx):
        sample = self.FB_data[idx] # no shuffling

        return sample

    def __len__(self):
        #len = self.data.shape[0]
        return self.len#self.data.shape[0]


class IFEDataset(IFEDataset_raw):
    """Iterable dataset implementation specific for IFE dataset"""
    #def __init__(self, merged_input_file, FB_data_input_file = None, F0_ref_input_file = None, min_value=None, max_value=None, maintain_data_continuity=False, super_chunk_size = 20 ** 6):
    def __init__(self, merged_input_file=None, FB_data_input_file=None, F0_ref_input_file=None, F0_ref=None, FB_data=None, FB_params = None, dataset_options=None, SNR = None):
        self.dataset_options = dataset_options
        self.labels_transformer_options = None
        self.SNR_dB = SNR

        # logging.info(dataset_options) <= will be done within base class initializer
        power_normalization = dataset_options["FB_data power normalization mode"]
        frequency_normalization = dataset_options["FB_data frequency normalization mode"]
        
        self.maintain_data_continuity = dataset_options["maintain data continuity"]
        self.do_shuffle = dataset_options["shuffle_data"]

        self.labels_transformer = None
        self.contains_labels = False

        self.FB_data = None
        # IFEDataset_raw.__init__(self, FB_data = FB_data, FB_params = FB_params, dataset_options=dataset_options, SNR = SNR)

        self.len = 0
        if merged_input_file == None:
            # determine length based on filesize in case of *.dat files
            self.F0_ref = None
            if F0_ref_input_file != None:
                F0_ref_input_file = PurePath(F0_ref_input_file)
                if F0_ref_input_file.suffix == ".dat":
                    logging.info("Reading *.dat F0_ref_input_file")
                elif F0_ref_input_file.suffix == ".csv":
                    logging.info("Reading *.csv F0_ref_input_file")
                F0_ref_len = self.__read_F0_ref__(F0_ref_input_file)
                F0_ref = self.F0_ref


            if FB_data_input_file != None:
                FB_data_input_file = PurePath(FB_data_input_file)
                if FB_data_input_file.suffix == ".dat":
                    logging.info("Reading *.dat FB_data_input_file")
                elif FB_data_input_file.suffix == ".csv":
                    logging.info("Reading *.csv FB_data_input_file")

            if FB_data_input_file is None:
                if F0_ref is not None:
                    # use synthesized data
                    self.F0_ref  = F0_ref
                    F0_ref_len = len(F0_ref)
                else:
                    self.F0_ref = None
                    F0_ref_len = -1


                # self.FB_data = FB_data
                IFEDataset_raw.__init__(self, FB_data = FB_data, FB_params = FB_params, dataset_options=dataset_options, SNR = SNR)
                FB_data_len = self.FB_data.shape[0]
                if FB_data_len < F0_ref_len:
                    # TODO correct data synthesis size
                    logging.error(f"FB_data_len({FB_data_len}) < F0_ref_len({F0_ref_len})")
                    F0_ref_len = FB_data_len
                    self.F0_ref = self.F0_ref[:F0_ref_len]
                
                self.FB_data_header = {}
                self.FB_data_header["Fc_max"] = FB_params.F_k_max
                self.FB_data_header["Fc_min"] = FB_params.F_0_min
            else:
                # TODO read in chunks? https://gist.github.com/f-huang/d2a949ecc37ec714e198c45498c0b779
                #            https://stackoverflow.com/questions/52209290/how-do-i-make-a-progress-bar-for-loading-pandas-dataframe-from-a-large-xlsx-file
                FB_data_header_size = self.__read_FB_data_header__(FB_data_input_file)
                if FB_data_header_size != -1:
                    if "FB_data_no_of_inputs" not in dataset_options:
                        dataset_options["FB_data_no_of_inputs"] = -1
                    read_data = self.__read_data__(F0_ref_input_file, FB_data_input_file, FB_data_header_size, FB_data_no_of_inputs=dataset_options["FB_data_no_of_inputs"])
                    IFEDataset_raw.__init__(self, FB_data = read_data["FB_data"], FB_params = FB_params, dataset_options=dataset_options, SNR = SNR)
                    F0_ref_len = read_data["F0_ref_len"]
                    FB_data_len = read_data["FB_data_len"]
                else:
                    F0_ref_len = 0
                    FB_data_len = 0

                    logging.info(dataset_options)
                    raise Exception("Incorrect FB_data_input_file (\"", FB_data_input_file, "\")")

            # determine data len
            if (F0_ref_len >= 0) and (F0_ref_len != FB_data_len):
                raise Exception("Cannot determine data len: labels_len=", F0_ref_len, ", FB_data_len=", FB_data_len)
            
            # # DATA NORMALIZATION
            # logging.info("FB_data normalization ...")
            # # 1. normalize power entries (first element and then every second)
            # #inst_powers = self.FB_data[:,0::2]
            # max_P = np.amax(self.FB_data[:,0::2],1) # start:stop:step
            # # https://www.delftstack.com/howto/numpy/python-numpy-divide-matrix-by-vector/
            # logging.info(f"Instantaneous power normalization: {power_normalization}")
            # if (power_normalization == "lin") or (power_normalization == "log"):
            #     self.FB_data[:,0::2] = self.FB_data[:,0::2] / max_P[:,None]
            # else:
            #     logging.error(f"Skipping instantaneous power normalization")
            # max_P = np.amax(self.FB_data[:,0::2],1) # start:stop:step

            # if power_normalization == "log":
            #     logging.info("Applying log...")
            #     # TODO add threshold to load_option
            #     self.FB_data[:,0::2][self.FB_data[:,0::2] < 0.000001] = 0.000001
            #     self.FB_data[:,0::2] = np.log(self.FB_data[:,0::2])
                
            # # 2. normalize frequencies
            # logging.info(f"Instantaneous frequency normalization: {frequency_normalization}")
            # if frequency_normalization == "lin_min":
            #     self.FB_data[:,1::2] = (self.FB_data[:,1::2]-self.FB_data_header["Fc_min"]) /(self.FB_data_header["Fc_max"]-self.FB_data_header["Fc_min"])
            # elif (frequency_normalization == "lin_0") or (frequency_normalization == "log"):
            #     self.FB_data[:,1::2] = self.FB_data[:,1::2] / self.FB_data_header["Fc_max"]
            # else:
            #     logging.error(f"Skipping instantaneous frequency normalization")

            # if frequency_normalization == "log":
            #     logging.info("Applying log...")
            #     # TODO add threshold to load_option
            #     self.FB_data[:,1::2][self.FB_data[:,1::2] < 0.000001] = 0.000001
            #     self.FB_data[:,1::2] = np.log(self.FB_data[:,1::2])

            # logging.info("FB_data normalization DONE")

            # # TODO add clipping

        else:
            merged_input_file = PurePath(merged_input_file)

            data_np = np.array(pd.read_csv(merged_input_file, header=None, dtype=np.float32))
            logging.info(r"read_csv DONE")
            #self.data_np = np.array(self.data)

            # split data and F0_ref
            # self.FB_data = data_np[:,:-1]
            IFEDataset_raw.__init__(self, FB_data = data_np[:,:-1], FB_params = FB_params, dataset_options=dataset_options, SNR = SNR)
            self.F0_ref = data_np[:,-1]

        
        # self.indexes = np.array(range(0,FB_data_len)) # indexes for data access
        self.indexes = np.array(range(0,len(self.FB_data))) # indexes for data access
        # Data discarding an shuffle 
        if (self.maintain_data_continuity == False) or (self.do_shuffle == True):
            # # 1. use: mask = tuple(F0_ref > 0)
            if self.F0_ref is not None:
                if self.maintain_data_continuity == False:
                    self.indexes = self.indexes[(self.F0_ref > 0)]

            # 2. shuffle index array and use it
            if self.do_shuffle:
                logging.info(f"Shuffling data indexes")
                self.indexes = self.indexes[np.random.permutation(len(self.indexes))]

            # self.F0_ref = self.F0_ref[indexes]
            # logging.info(f"Shuffling F0_ref DONE")
            # self.F0_labels = self.F0_labels[indexes]
            # logging.info(f"Shuffling F0_labels DONE")
            # self.FB_data = self.FB_data[indexes,:]
            # logging.info(f"Shuffling FB_data DONE")

        self.len = len(self.indexes)


    def convert_labels(self, labels_transformer_options):
        self.labels_transformer_options = labels_transformer_options

        self.classes_edges = None
        self.classes_labels = None
        
        use_F0_too_low_class = labels_transformer_options["use_F0_too_low_class"]
        if labels_transformer_options["mode"] is not None:
            F0_ref_min_value = labels_transformer_options["F0_ref_min"]
            F0_ref_max_value = labels_transformer_options["F0_ref_max"]
            no_of_classes = labels_transformer_options["no_of_classes"]

            # prepare for labels transforming
            if labels_transformer_options["mode"] == "lin":
                self.classes_labels = ec.get_lin_edges(F0_ref_min_value, F0_ref_max_value, no_of_classes)
                dF0_ref = self.classes_labels[1] - self.classes_labels[0]
                self.classes_edges = self.classes_labels - dF0_ref/2
                self.classes_edges = np.append(self.classes_edges, self.classes_labels[-1] + dF0_ref/2)
                linear_scale = True
            elif labels_transformer_options["mode"] == "log":
                self.classes_labels = ec.get_log_edges(F0_ref_min_value, F0_ref_max_value, no_of_classes)
                dF0_ref_factor = np.sqrt(self.classes_labels[1] / self.classes_labels[0])
                self.classes_edges = self.classes_labels / dF0_ref_factor
                self.classes_edges = np.append(self.classes_edges, self.classes_labels[-1] * dF0_ref_factor)
                linear_scale = False

            if use_F0_too_low_class:
                self.classes_labels = [np.float32(x) for x in np.insert(self.classes_labels, 0, 0.0)]
            else:
                self.classes_labels = [np.float32(x) for x in self.classes_labels]
            self.encoder = LabelEncoder()
            self.encoder.fit(self.classes_labels)

            if (not use_F0_too_low_class) and (not self.maintain_data_continuity):
                # add class at 0.0 just for discarding unfit data
                self.classes_labels = [np.float32(x) for x in np.insert(self.classes_labels, 0, 0.0)]
            
            self.labels_transformer = LabelsTransformer(labels_path = None, 
                min_value = F0_ref_min_value, max_value = F0_ref_max_value, 
                classes_labels = self.classes_labels, classes_edges = self.classes_edges, 
                classes_number = no_of_classes, use_F0_too_low_class=((not self.maintain_data_continuity) or use_F0_too_low_class), 
                linear_scale=linear_scale) # TODO linear_scale => add as parameter to configs

        if self.labels_transformer is not None:
            # labels transforming
            #data_np[:,-1] = self.encoder.transform(data_np[:,-1])
            if self.F0_ref is not None:
                logging.info(r"labels encoding...")
                F0_ref_edges = np.float32(self.labels_transformer._transform_nparray_within_edges(self.F0_ref))
        
                # may contain zeros 
                self.indexes = np.array(range(0,len(self.F0_ref))) # indexes for data access
                if self.maintain_data_continuity == False:
                    self.indexes = self.indexes[F0_ref_edges > 0]
                
                self.F0_labels = np.zeros_like(F0_ref_edges, dtype=np.int16)
                # encode only relevant values
                self.F0_labels[self.indexes] = self.encoder.transform(F0_ref_edges[self.indexes])

                logging.info(r"labels encoding DONE")
                self.contains_labels = True
            else:
                logging.info(r"F0_ref is None => No labels")
                self.F0_labels = None
                self.contains_labels = False
        else:
            self.F0_labels = self.F0_ref
            self.contains_labels = False

        # Data discarding and shuffle 
        # self.indexes = np.array(range(0,len(self.F0_ref))) # indexes for data access
        self.indexes = np.array(range(0,len(self.FB_data))) # indexes for data access
        if (self.maintain_data_continuity == False) or (self.do_shuffle == True):
            # # 1. use: mask = tuple(F0_ref > 0)
            if (not self.maintain_data_continuity) and (use_F0_too_low_class):
                # only when use_F0_too_low_class == True there is additional class 0 with label 0
                self.indexes = self.indexes[(self.F0_labels > 0)]

            # 2. shuffle index array and use it
            if self.do_shuffle:
                logging.info(f"Shuffling data indexes")
                self.indexes = self.indexes[np.random.permutation(len(self.indexes))]

            # self.F0_ref = self.F0_ref[indexes]
            # logging.info(f"Shuffling F0_ref DONE")
            # self.F0_labels = self.F0_labels[indexes]
            # logging.info(f"Shuffling F0_labels DONE")
            # self.FB_data = self.FB_data[indexes,:]
            # logging.info(f"Shuffling FB_data DONE")

        self.len = len(self.indexes)
        
    def __read_FB_data_header__(self, FB_data_file, header_version = 0):
        # read FB_data_input_file header
        self.FB_data_header = {}
        header_size = -1
        if FB_data_file.suffix == ".dat":
            logging.info("Reading *.dat FB_data_input_file header")
            with open(FB_data_file, 'rb') as f:
                # make something sensible here
                if header_version == 0:
                    header_len = np.fromfile(f, dtype=np.short, count=1)[0]
                    header_size = (1+header_len) * np.dtype(np.short).itemsize # count in also the header length field
                    if header_len != 8:
                        raise Exception("Wrong FB_data file header size")
                    buffer = np.fromfile(f, dtype=np.short, count=header_len)
                    self.FB_data_header = {
                        "FB_K" : buffer[0], # number of filter
                        "FB_N_out" : buffer[1], # number of outputs per filter
                        "data_length" : buffer[2], # TODO correct header: change - too small value range
                        "F_s" :  buffer[3],  # TODO correct header: change - too small value range
                        "Fc_min" : buffer[4],
                        "Fc_max" : buffer[5],
                        "bins_per_octave" : buffer[6], # TODO correct header: wrong type
                        "q" : buffer[7] # TODO correct header: wrong type
                    }
                else:
                    logging.info(f"not implemented for header version:{header_version}")
                    # reading header failed
            logging.info(self.FB_data_header)
        else:
            logging.info("Reading *.csv file: no header to read")
            
        # successful header reading 
        return header_size  # -1 on failure

    def __read_F0_ref__(self, F0_ref_file):
        # read F0
        if F0_ref_file.suffix == ".dat":
            logging.info("Reading *.dat F0_ref_file")
            # no header just F0 data
            with open(F0_ref_file, 'rb') as f:
                self.F0_ref = np.fromfile(f, dtype=np.float32, count=-1)
        else:
            # read from csv
            logging.info("Reading *.csv F0_ref_file")
            # self.F0_ref = np.array(pd.read_csv(F0_ref_file, header=None, dtype=np.float32)) # problem: need to be converted into vector
            # self.F0_ref = np.loadtxt(F0_ref_file, dtype=np.float32) # alternative but slower
            self.F0_ref = pd.read_csv(F0_ref_file, header=None, dtype=np.float32).to_numpy()
            self.F0_ref = self.F0_ref.flatten()

        F0_ref_len = self.F0_ref.shape[0]
        return F0_ref_len

    def __read_data__(self, F0_ref_file, FB_data_file, FB_data_header_size = 0, FB_data_no_of_inputs = -1):
        # read F0
        if F0_ref_file is not None:
            F0_ref_len = self.__read_F0_ref__(F0_ref_file)
        
        # read FB data
        if FB_data_file.suffix == ".dat":
            logging.info("Reading *.dat FB_data_file")
            # skip header and read just F0 data
            with open(FB_data_file, 'rb') as f:
                # TODO get file size and then read multiple of self.FB_data_header["FB_K"] * self.FB_data_header["FB_N_out"]
                # self.FB_data = np.fromfile(f, dtype=np.float32, count=super_chunk_size * self.FB_data_header["FB_K"] * self.FB_data_header["FB_N_out"])
                FB_data = np.fromfile(f, dtype=np.float32, count=-1, offset = FB_data_header_size)
                FB_data = np.reshape(FB_data, [-1, self.FB_data_header["FB_K"] * self.FB_data_header["FB_N_out"]])
        else:
            # read from csv FD_data file
            data_np = np.array(pd.read_csv(FB_data_file, header=None, dtype=np.float32))
            logging.info(r"FB_data read_csv DONE")

            # split data and F0_ref
            FB_data = data_np[:,0:FB_data_no_of_inputs]

        # TODO check size of read data to determine if it was the last super_chunk
        FB_data_len = self.FB_data.shape[0]
        logging.info(f"self.FB_data.shape[0]:{FB_data_len}")

        return {"F0_ref_len": F0_ref_len, "FB_data_len": FB_data_len, "FB_data": FB_data}

    @staticmethod
    def load_classes_labels(load_path):
        # just returns classes labels loaded from file
        if len(load_path.as_posix()) > 0:
            filename = load_path / "infer_classes_labels"
        else:
            raise Exception("save_epoch_infer_data: empty load_path")

        json_filename = Path(filename).with_suffix(".json")
        if json_filename.is_file():
            with open(json_filename,'r+') as f:
                loaded_classes = json.load(f)
                return loaded_classes["classes_labels"]

        return None

    def save_classes_labels(self, save_path):
        if len(save_path.as_posix()) > 0:
            filename_labels = save_path / "infer_classes_labels"
            filename_F0_ref_indexes = save_path / "infer_F0_ref_indexes"
        else:
            raise Exception("save_epoch_infer_data: empty save_path")

        Path(save_path).mkdir(parents=True, exist_ok=True)

        # classes_filename = Path(filename).with_suffix(".npz")
        # if classes_filename.is_file():
        #     loaded_classes = np.load(classes_filename)
        #     res1 = np.array_equal(self.classes_labels, loaded_classes["classes_labels"])

        #     if res1:
        #         logging.info("skipping save_classes_labels: classes_labels file already saved")
        #     else:
        #         raise Exception(f"save_classes_labels: file {classes_filename} content differs from current classes labels")

        # np.savez_compressed(classes_filename, classes_labels=self.classes_labels) 

        if Path(filename_F0_ref_indexes).with_suffix(".npz").is_file():
            loaded = np.load(Path(filename_F0_ref_indexes).with_suffix(".npz"))
            res2 = np.array_equal(self.indexes, loaded["F0_ref_indexes"])

            if res2:
                logging.info("skipping: F0_ref_indexes file already saved")
            else:
                full_filename = Path(filename_F0_ref_indexes).with_suffix(".npz")
                raise Exception(f"save_classes_labels: file {full_filename} content differs from given F0_ref_indexes")

        else:
            # np.savez(filename, F0_ref=self.F0_ref, indexes=self.indexes)
            np.savez_compressed(filename_F0_ref_indexes, F0_ref_indexes=self.indexes) # test

        json_filename = Path(filename_labels).with_suffix(".json")
        if json_filename.is_file():
            with open(json_filename,'r+') as f:
                loaded_classes = json.load(f)
                res1 = np.array_equal(self.classes_labels, loaded_classes["classes_labels"])
                res2 = loaded_classes["maintain_data_continuity"] == self.maintain_data_continuity

                if res1 and res2:
                    logging.info("skipping save_classes_labels: classes_labels file already saved")
                    return
                else:
                    raise Exception(f"save_classes_labels: file {json_filename} content differs from current classes labels")

        with open(json_filename,'w+') as f:
            data = {"classes_labels": [float(x) for x in self.classes_labels],
                    "maintain_data_continuity": self.maintain_data_continuity}
            json.dump(data, f, indent=4)            

    def save_F0_ref(self, save_path, validation_data_options):
        if len(save_path.as_posix()) > 0:
            # if (save_path[-1]  == '\\') or (save_path[-1]  == '/'):
            #     filename = save_path + "infer_F0_ref"
            # else:
            #     filename = save_path + "/infer_F0_ref"
            filename = save_path / "infer_F0_ref"
        else:
            raise Exception("save_epoch_infer_data: empty save_path")

        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        if Path(filename).with_suffix(".npz").is_file():
            loaded = np.load(Path(filename).with_suffix(".npz"))
            res1 = np.array_equal(self.F0_ref, loaded["F0_ref"])

            if res1:
                logging.info("skipping save_F0_ref: F0_ref file already saved")
            else:
                full_filename = Path(filename).with_suffix(".npz")
                raise Exception(f"save_F0_ref: file {full_filename} content differs from given F0_ref")

        else:
            # np.savez(filename, F0_ref=self.F0_ref, indexes=self.indexes)
            np.savez_compressed(filename, F0_ref=self.F0_ref) # test

            # TODO save also
            validation_data_options.pop("skip", None)
            with open(Path(filename).with_suffix(".json"),'w+') as f:
                json.dump(validation_data_options, f, indent=4)            
        


    def __getitem__(self, idx):
        index = self.indexes[idx] # allows shuffling without moving data in memory and keeping original data
        sample = self.FB_data[index]
        if self.F0_labels is None:
            label = np.nan
        else:
            label = self.F0_labels[index]
        if self.F0_ref is None:
            ref_value = np.nan
        else:
            ref_value = self.F0_ref[index]

        return sample, label, ref_value

    # def __getitem__(self, idx):
    #     # dd = self.data_np
    #     row = self.data_np[idx]
    #     sample = row[:-2]
    #     ref_value = row[-2]
    #     label = row[-1]

    #     return sample, label, ref_value

    def __len__(self):
        #len = self.data.shape[0]
        return self.len#self.data.shape[0]

