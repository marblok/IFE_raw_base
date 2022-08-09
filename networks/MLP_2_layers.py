import torch.nn as nn
import logging
from collections import OrderedDict
from math import floor
class Model_auxdata:
    def __init__(self):
        self.training_data_options = None
        self.dataset_options = None
        self.labels_transformer_options = None
        self.trainer_config = None

    @staticmethod
    def compare_model_configs(model_config, model_config_ref):
        # "model_config": {
        #     "type": "MLP_2_layers", 
        #     "parameters": {
        #     },
        #     "no_of_inputs": 40,
        #     "no_of_classes": 100
        # }
        if "type" not in model_config_ref:
            logging.error("No type in model_config_ref")
            return False

        # check type
        if model_config["type"] != model_config_ref["type"]:
            logging.error("Incompatible model types")
            return False

        # check basic config
        if model_config["no_of_inputs"] != model_config_ref["no_of_inputs"]:
            logging.error("Incompatible model no_of_inputs")
            return False
        if model_config["no_of_classes"] != model_config_ref["no_of_classes"]:
            logging.error("Incompatible model no_of_classes")
            return False
        
        # check basic parameters (per type)
        param = model_config["parameters"]
        param_ref = model_config_ref["parameters"]
        if model_config["type"] == "MLP_2_layers":
            # no model parameters expected
            return True
        elif model_config["type"] == "MLP_multilayer":
            if param == param_ref:
                return True
            else:
                logging.error("MLP_multilayer model parameters mismatch")
                logging.error(f"Found:{param}")
                logging.error(f"Expected:{param_ref}")
            input("Press Enter")
            return False
        elif model_config["type"] == "MLP_3_layers":
            logging.error("MLP_3_layers model type is not supported yet")
            input("Press Enter")
            return False
        else:
            logging.error("Unknown model type")
            input("Press Enter")
            return False

    def store_options(self, training_data_options, dataset_options, labels_transformer_options, trainer_config, model_config):
        self.training_data_options = training_data_options
        self.dataset_options = dataset_options
        self.labels_transformer_options = labels_transformer_options
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.tag_str = Model_auxdata.get_tag_from_options(training_data_options, dataset_options, labels_transformer_options, trainer_config, model_config)

    @staticmethod
    def get_tag_from_options(training_data_options, dataset_options, labels_transformer_options, trainer_config, model_config):
        tag_str =  str(dataset_options["FB_data power normalization mode"])
        tag_str += "_" + str(dataset_options["FB_data frequency normalization mode"])
        tag_str += "_" + str(labels_transformer_options["mode"])
        tag_str += "_N=" + str(model_config["no_of_classes"])
        tag_str += "_" + str(labels_transformer_options["use_F0_too_low_class"])
        if "tag" in training_data_options:
            tag_str += "_" + str(training_data_options["tag"])
        lr = trainer_config["learning_rate"]
        tag_str += f"_lr={lr:.0e}"
        return tag_str


class MLP_2_layers(nn.Module, Model_auxdata):
    """Defines networks topology based on number of input examples and output classes"""
    def __init__(self, n_inputs, n_classes, use_F0_too_low_class=False):
        super(MLP_2_layers, self).__init__()
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.use_F0_too_low_class = use_F0_too_low_class
        if self.use_F0_too_low_class:
            self.n_classes += 1
        self.tag_str = ""

        self.model_layers = nn.Sequential(
            nn.Linear(self.n_inputs, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, self.n_classes)
        )

    def forward(self, x):
        """Defines forward run through the network's topology"""
        x = self.model_layers(x)
        return x

class MLP_multilayer(nn.Module, Model_auxdata):
    """Defines networks topology based on number of input examples and output classes"""
    def __init__(self, n_inputs, n_classes, layers_defs, use_F0_too_low_class=False):
        super(MLP_multilayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.use_F0_too_low_class = use_F0_too_low_class
        if self.use_F0_too_low_class:
            self.n_classes += 1
        self.tag_str = ""

        self.use_unsqueeze = False
        modules = OrderedDict()
        key_ind = 0
        previous_n_out = self.n_inputs # L_out
        L_out = previous_n_out
        for layer_def in layers_defs:
            key_str = str(key_ind)
            if layer_def[0] == "Linear":
                n_outputs = layer_def[1]
                if n_outputs is None:
                    modules[key_str] = nn.Linear(previous_n_out, self.n_classes)
                else:
                    modules[key_str] = nn.Linear(previous_n_out, n_outputs)
                previous_n_out = n_outputs
                L_out = previous_n_out
                key_ind += 1
            elif layer_def[0] == "Conv1d":
                self.use_unsqueeze = True
                C_input = layer_def[1]
                C_output = layer_def[2]
                kernel_size = layer_def[3]
                stride = layer_def[4]
                padding = 0
                dilation = 1
                if C_output is None:
                    # TODO what should be put into out_channels
                    modules[key_str] = nn.Conv1d(in_channels=C_input, out_channels=self.n_classes/C_input, kernel_size=kernel_size, stride=stride)
                else:
                    modules[key_str] = nn.Conv1d(in_channels=C_input, out_channels=C_output, kernel_size=kernel_size, stride=stride)
                
                L_out = floor((L_out+2*padding-dilation*(kernel_size-1)-1)/stride+1)
                previous_n_out = C_output*L_out
                key_ind += 1
            elif layer_def[0] == "Flatten":
                modules[key_str] = nn.Flatten()
                L_out = previous_n_out
                key_ind += 1
            elif layer_def[0] == "ReLU":
                modules[key_str] = nn.ReLU()
                key_ind += 1
            elif layer_def[0] == "LeakyReLU":
                negative_slope = layer_def[1]
                modules[key_str] = nn.LeakyReLU(negative_slope)
                key_ind += 1
            elif layer_def[0] == "Tanh":
                modules[key_str] = nn.Tanh()
                key_ind += 1
            elif layer_def[0] == "Hardtanh":
                modules[key_str] = nn.Hardtanh()
                key_ind += 1

        if (previous_n_out != self.n_classes) and (previous_n_out is not None):
            raise Exception(f"Last layer outpits number ({previous_n_out}) differes from self.n_classes({self.n_classes})")
        self.model_layers = nn.Sequential(modules)

    def forward(self, x):
        """Defines forward run through the network's topology"""
        if self.use_unsqueeze:
            x = x.unsqueeze(1) # reshape the input for CNN layer
        x = self.model_layers(x)
        return x

class MLP_3_layers(nn.Module, Model_auxdata):
    """Defines networks topology based on number of input examples and output classes"""
    def __init__(self, n_inputs, n_classes, bottle_neck_size = 40, use_F0_too_low_class=False):
        super(MLP_3_layers, self).__init__()
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.use_F0_too_low_class = use_F0_too_low_class
        if self.use_F0_too_low_class:
            self.n_classes += 1
        self.tag_str = ""

        self.model_layers = nn.Sequential(
            nn.Linear(self.n_inputs, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, bottle_neck_size),
            nn.ReLU(),
            nn.Linear(bottle_neck_size, self.n_classes)
        )

    def forward(self, x):
        """Defines forward run through the network's topology"""
        x = self.model_layers(x)
        return x


