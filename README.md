# HSI
HSI Neural Networks

[1] Blok, M., Banaś, J., & Pietrolaj, M. (2021, July). IFE: NN-aided Instantaneous Pitch Estimation. In 2021 14th International Conference on Human System Interaction (HSI) (pp. 1-7). IEEE. 

[2] Blok, M., Banaś, J., & Pietrołaj, M. (2022). Strategie treningu neuronowego estymatora częstotliwości tonu krtaniowego z użyciem generatora syntetycznych samogłosek. Przegląd Telekomunikacyjny+ Wiadomości Telekomunikacyjne, 604-607.

**NOTE:** The below instructions are outdated and relate to the previous versions of the project.
The latest instructions (in Polish) are available on projects Wiki page: https://git.pg.edu.pl/ife/IFE_raw_base/-/wikis/home

# Requirements & installation:
1. Install Miniconda3 for Windows: https://docs.conda.io/en/latest/miniconda.html
2. Create new conda environment:
	- use command: <i>conda create -n \<your-env-name> python=3.9</i>
	- activate it: <i>conda activate \<your-env-name></i>
3. Install Pytorch on your active environment:
      - use command: <i>conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch</i>
4. Install additional python modules with requirement.txt file from this repository:
      - use command: <i>conda install -c conda-forge --file requirements.txt</i>

Once the above steps are completed, the environment is ready to be run.

 # Project overview

There are 3 main scripts that can be run in order to execute IFE related experiment.
The below files can be treated as examples of proper usage of the scripts:

1. <b>prepare_data.py</b> - transforms input data to a format that can be used by IFE training and evaluation steps.
		It takes data from a given .dat file and labels from .csv file. Dat2CSV normalizes data and creates .csv file with data. Then the steps transform labels to a given min/max range and number of output classes. At the end both normalized data and transformed labels are combined together to create input format compatible for training and inference steps.

<u>Arguments</u>:
<i>
- data_dat_path - path to .dat file with training or validation data
- labels_csv_path - path to .csv with labels matching the data from data_dat_path file
- final_data_output_path - path where the final output of data preparation will be saved
- maintain_data continuity - is set to True, skips shuffling and zeroes removal to make input signal & data comparison easier - should be used only for evaluation data
</i>
<br>
2. <b>train_model.py</b> - trains given network topology with provided input data (its format must match the one created by prepare_data.py script). This step reads the provided data to a data_loader and transform original labels to class numbers (encoding). Once the training is finished all related collaterals are saved to "output_data\models" directory. Epoch with the best accuracy is saved as "BEST_EPOCH". Not all model & training parameters are taken as function arguments. In order to maintain consistency between different steps of the experiment a separate file with common constants has been created under "utils\constants.py". Parameters like input & output sizes, epochs number and learning rate, can be changed there. In case a new model topology is needed, it can be created under "networks" directory and then used in train_model.py execution. "networks\MLP_2_layers.py" can be used as an example.

<u>Arguments</u>:
<i>
- model - neural network model topology that will be used for training
- train_data_path - path to .csv file with training data (its format must match the one created by prepare_data.py script).
</i>
<br>
3. <b>infer_model.py</b> - runs inference/evaluation on an already trained network model. It displays accuracy achieved on a given dataset and saves predictions and decoded original labels to "output_data/predictions". Additionally this steps saves and displays a plot with comparison between original labels values and predictions given by the model.

<u>Arguments</u>:
<i>
- model - neural network model topology that will be used for inference
- model_path - path to an already trained model matching topology of the previous "model" argument. This one will be used to get inference results.
- input_data_path - path to .csv file with evaluation data (its format must match the one created by prepare_data.py script).
</i>

# Usage

To run the chosen script chose execute it under previously created (and active) conda environment. Use command: <i>python \<script-name></i>example: <i>python prepare_data.py</i>. All paths should be set directly in the script file.
