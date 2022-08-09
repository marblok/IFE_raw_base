from pathlib import PurePath
import pandas as pd
import numpy as np
from numpy.lib.scimath import log10
from tqdm import tqdm
from datetime import datetime
import logging
# import glob

from data_processor import edges_calculators as ec

class LabelsTransformer():
    def __init__(self, labels_path, min_value, max_value, classes_labels, classes_edges, classes_number, use_F0_too_low_class=False, linear_scale=True):
        self.labels_path = labels_path
        self.min_value = min_value
        self.max_value = max_value
        self.linear_scale = linear_scale
        #self.maintain_data_continuity = maintain_data_continuity
        self.use_F0_too_low_class = use_F0_too_low_class

        self.classes_edges = classes_edges
        self.classes_labels = classes_labels
        # if (classes_edges is not None) and (classes_labels is not None):
        #     self.classes_edges = classes_edges
        #     self.classes_labels = classes_labels
        # else:
        #     if linear_scale:
        #         logging.info(f"Using linear scale for labels.")
        #         self.edges = ec.get_lin_edges(self.min_value, self.max_value, classes_number)
        #     else: # log
        #         logging.info(f"Using log scale for labels.")
        #         self.edges = ec.get_log_edges(self.min_value, self.max_value, classes_number)

    @staticmethod
    def compare_labels_configs(labels_transformer_options, labels_transformer_options_ref):
        # "labels transformer options": {
        #     "mode": "lin", 
        #     "F0_ref_min": 50, 
        #     "F0_ref_max": 400, 
        #     "use_F0_too_low_class": true
        # },

        if "mode" not in labels_transformer_options_ref:
            logging.info("No mode in labels_transformer_options_ref")
            return False

        # check mode
        if labels_transformer_options["mode"] != labels_transformer_options_ref["mode"]:
            logging.error("Incompatible label transformer modes")
            return False

        # check basic config
        if labels_transformer_options["F0_ref_min"] != labels_transformer_options_ref["F0_ref_min"]:
            logging.error("Incompatible label transformer F0_ref_min")
            return False
        if labels_transformer_options["F0_ref_max"] != labels_transformer_options_ref["F0_ref_max"]:
            logging.error("Incompatible label transformer F0_ref_max")
            return False
        if labels_transformer_options["use_F0_too_low_class"] != labels_transformer_options_ref["use_F0_too_low_class"]:
            logging.error("Incompatible label transformer use_F0_too_low_class")
            return False

        return True

    # def _transform_within_edges(self, input_number):
    #     if input_number == 0:
    #         if self.maintain_data_continuity:
    #             return 0
    #         else:
    #             return self.edges[0]
    #     elif input_number <= self.min_value:
    #         return self.edges[0] # return min
    #     elif input_number >= self.max_value:
    #         return self.edges[-1] # return max
    #     else:
    #         for i in range(0, len(self.edges)):
    #             if self.edges[i] == input_number:
    #                 return self.edges[i]
    #             elif input_number > self.edges[i] and input_number < self.edges[i+1]: # add rounding to nearest value
    #                 if(abs(input_number - self.edges[i]) < abs(input_number - self.edges[i+1])):
    #                     return self.edges[i]
    #                 else:
    #                     return self.edges[i+1]

    #     logging.error(f"The given value could not be resolved during transformation withing provided edges: {input_number}")

    # def _transform_nparray_within_edges(self, input_numbers):
    #     if self.linear_scale == False:
    #         raise Exception("log labels not supported yet")

    #     output_numbers  = np.copy(input_numbers) # TODO check if this is neccessary

    #     # output_numbers[input_numbers <= self.min_value] = self.edges[0]
    #     # output_numbers[input_numbers >= self.max_value] = self.edges[-1] # return max

    #     output_numbers[input_numbers <= (self.edges[0]+self.edges[1])/2] = self.edges[0]
    #     if self.use_F0_too_low_class:
    #         output_numbers[input_numbers <= self.edges[0]-(self.edges[1]-self.edges[0])/2] = 0.0 # F0 too low class
    #     for i in range(1, len(self.edges)-1):
    #         #output_numbers[self.edges[i] == input_numbers] = self.edges[i]
    #         output_numbers[(input_numbers > (self.edges[i]+self.edges[i-1])/2) & (input_numbers <= (self.edges[i]+self.edges[i+1])/2)] = self.edges[i]

    #     output_numbers[input_numbers > (self.edges[-1]+self.edges[-2])/2] = self.edges[-1] # return max
    #     # TODO verify if it won't be faster to check all entries one by one

    #     return output_numbers

    def _transform_nparray_within_edges(self, input_numbers):
        output_numbers  = np.copy(input_numbers) # TODO check if this is neccessary

        labels_offset = 0
        if self.use_F0_too_low_class:
            labels_offset = 1

        output_numbers[input_numbers <= self.classes_edges[1]] = self.classes_labels[0 + labels_offset]
        if self.use_F0_too_low_class:
            output_numbers[input_numbers <= self.classes_edges[0]] = self.classes_labels[0] # F0 too low class
        for i in range(1, len(self.classes_edges)-1):
            # output_numbers[(input_numbers > (self.edges[i]+self.edges[i-1])/2) & (input_numbers <= (self.edges[i]+self.edges[i+1])/2)] = self.edges[i]
            output_numbers[(input_numbers > self.classes_edges[i]) & (input_numbers <= self.classes_edges[i+1])] = self.classes_labels[i + labels_offset]

        #output_numbers[input_numbers > (self.edges[-1]+self.edges[-2])/2] = self.edges[-1] # return max
        output_numbers[input_numbers > self.classes_edges[-1]] = self.classes_labels[-1] # return max
        # TODO verify if it won't be faster to check all entries one by one

        return output_numbers

    # def _transform_ints_within_edges(self, input_number):
    #     # TODO make compatible with _transform_nparray_within_edges

    #     if input_number == 0:
    #         if not self.maintain_data_continuity:
    #             return 0
    #         else:
    #             return self.edges[0]
    #     elif input_number <= self.min_value:
    #         return self.edges[0] # return min
    #     elif input_number >= self.max_value:
    #         return self.edges[-1] # return max
    #     else:
    #         for i in range(0, len(self.edges)):
    #             if self.edges[i] == input_number:
    #                 return self.edges[i]
    #             elif input_number > self.edges[i] and input_number < self.edges[i+1]: # add rounding to nearest value
    #                 if(abs(input_number - self.edges[i]) < abs(input_number - self.edges[i+1])):
    #                     return self.edges[i]
    #                 else:
    #                     return self.edges[i+1]

    #     logging.error(f"The given value could not be resolved during transformation withing provided edges: {input_number}")


    def run(self):

        results = []

        logging.info(f"Reading labels data from: {self.labels_path}") # Read the file with raw labels
        data = pd.read_csv(self.labels_path, header=None, dtype=np.float32)
        data_np = data[data.columns[-1]].to_numpy()
        # data_np = np.rint(data_np).astype(int)

        logging.info("Processing labels data...") # Transform each value to be aligned with class edges
        # for value in tqdm(data_np):
        #     result = self._transform_ints_within_edges(value)
        #     results.append(result) # Transfrom to integers
        results = self._transform_nparray_within_edges(data_np)

        pd_results = pd.DataFrame(results, dtype=np.float32)

        save_path = PurePath(f"output_data/transformed_labels/{self.labels_path.stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
        logging.info(f"Saving labels data to: {save_path}")
        pd_results.to_csv(save_path, header=None, index=False)

        return save_path

#lin_labels_transformer = LabelsTransformer(PurePath("input_data/raw_data/training/15000events_50-400Hz_length500ms_gaps0ms/normalized_0_1/15000events_50-400Hz_length500ms_gaps0ms.csv"), min_value=50, max_value=400, classes_number=100)
#lin_labels_transformer.run()

#target_pattern = r"input_data\raw_data\validation\speech_data\normalized_0_1\labels\*.csv"
#paths = glob.glob(target_pattern)

#for path in paths:
#    lin_labels_transformer = LabelsTransformer(PurePath(path), min_value=50, max_value=400, classes_number=100)
#    lin_labels_transformer.run()