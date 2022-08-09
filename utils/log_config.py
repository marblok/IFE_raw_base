from datetime import datetime
import sys
import logging
from pathlib import Path, PurePath

class LogConfig:
    def __init__(self):
        self.additional_logging_file_handler = None
        self.log_filename = None

    def init_logging(self, module_name = None, user_output_folder = None):
        # logging.basicConfig(filename='infer_model_'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.log', level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logFormatter_file = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        logFormatter_std  = logging.Formatter("[%(levelname)-5.5s]  %(message)s")

        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)
        rootLogger.handlers.clear()

        if module_name is not None:
            log_filename= module_name + '_'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.log'

            # create directory for saving data
            if user_output_folder is None:
                log_path = Path(f"output_data/logs")
            else:
                log_path = Path(user_output_folder)
            log_path.mkdir(parents=True, exist_ok=True)

            # fileHandler = logging.FileHandler(f"output_data/logs/{log_filename}")
            full_log_filename = PurePath(log_path, log_filename)
            fileHandler = logging.FileHandler(full_log_filename.as_posix())
            fileHandler.setFormatter(logFormatter_file)
            rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter_std)
        rootLogger.addHandler(consoleHandler)


    def reinit_additional_output_file(self, root_folder = None, log_name = None):
        # if root_folder just removes additional logging handler

        rootLogger = logging.getLogger()
        if self.additional_logging_file_handler is not None:
            rootLogger.removeHandler(self.additional_logging_file_handler)

        if root_folder is not None:
            Path(root_folder).mkdir(parents=True, exist_ok=True)

            log_filename= log_name + '_'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.log'
            self.additional_logging_file_handler = logging.FileHandler(Path(root_folder, log_filename))

            logFormatter_file = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
            self.additional_logging_file_handler.setFormatter(logFormatter_file)
            rootLogger.addHandler(self.additional_logging_file_handler)
        