from engine.trainer import train_model
from utils.log_config import LogConfig

from pathlib import PurePath
import logging
import sys
import json

log_cfg = LogConfig()
log_cfg.init_logging("train_model")

#logging.info(f'{PurePath(__file__).name} started')
logging.info(f'{PurePath(sys.argv[0]).name} started with arguments: {sys.argv[1:]}')

if len(sys.argv) <= 1:
    logging.error(f'{PurePath(sys.argv[0]).name} requires config json file(s) as argument(s)')
    exit()
else:
    for argument in sys.argv[1:]:
        logging.info(f"Loading config from: {argument}")
        # TODO check if file has been successfuly open and read
        with open(argument, 'rt') as f:
            trainer_configuration_data = json.load(f)
            logging.info(trainer_configuration_data)

        # process loaded config data
        training_data_configs = trainer_configuration_data["training_data"]

        #for training_data_file in training_data_options["files"]:
        for training_data_options in training_data_configs:
            logging.info("Training data option:")
            logging.info(training_data_options)

            if not "skip" in training_data_options:
                training_data_options["skip"] = False
            if training_data_options["skip"] == True:
                logging.info("Skipping training_data_options entry")

            else:
                if not "reload" in training_data_options:
                    training_data_options["reload"] = 0

                train_dataset = None # force data reload
                for dataset_options in trainer_configuration_data["dataset_options"]:
                    # loaded_options basic info
                    logging.info("Dataset options:")
                    logging.info(dataset_options)

                    if not "skip" in dataset_options:
                        dataset_options["skip"] = False
                    if dataset_options["skip"] == True:
                        logging.info("Skipping config")
                    else:

                        for trainer_options in trainer_configuration_data["trainer_options"]:
                            logging.info("Trainer options:")
                            logging.info(trainer_options)

                            if not "skip" in trainer_options:
                                trainer_options["skip"] = False
                            if trainer_options["skip"] == True:
                                logging.info("Skipping config")
                            else:
                                # run training
                                logging.info("Starting training engine")
                                train_dataset = train_model(train_dataset, training_data_options, dataset_options, trainer_options, log_cfg) # runing training engine

    



