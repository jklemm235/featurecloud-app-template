from logic import fl_algorithm
from helper.run_app_simulation import run_simulation_featurecloud, run_simulation_native

inputfolders = ["inputfolder1", "inputfolder2"]
outputfolders = ["outputfolder1", "outputfolder2"]
generic_dir = "generic_dir"
fl_algorithm_function = fl_algorithm

### FEATURECLOUD SIMULATION (WITH DOCKER)
# This will run a federated learning simulation via FeatureCloud's Testing module.
# The data path will be given to the Featurecloud controller and
# the clientnames and generic_dir will be given to the test.
# the clientnames and generic_dir should be in the data_path and be given
# as relative paths to the data_path.
run_simulation_featurecloud(
    data_path="data",
    clientnames=inputfolders,
    generic_dir=generic_dir
)

### NATIVE SIMULATION (WITHOUT DOCKER)
# This will run a federated learning simulation natively, without using Docker.
# This means make sure to install all requirements you specified in requirements.txt!
run_simulation_native(
    clientpaths=inputfolders,
    outputfolders=outputfolders,
    generic_dir=generic_dir,
    fl_algorithm_function=fl_algorithm_function
)
