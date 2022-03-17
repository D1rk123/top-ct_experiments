# top-ct_experiments

This repository contains code to perform TOP-CT (Trajectory with Overlapping Projections x-ray Computed Tomography) experiments. The class MultiOperator can be used for experiments in general. Moreover three experiments are included from the work in progress paper on TOP-CT: carousel\_simulation\_experiment.py is the first experiment of that paper, attenuation\_tuning\_experiment.py is the second experiment and mandarin\_carousel\_experiment.py is the third experiment.


## Running the code
To clone the repository with submodules use the following command:
```
git clone --recurse-submodules git@github.com:D1rk123/top-ct_experiments.git
```

To run the scripts you need to create an extra script *folder_locations.py* that contains two functions: get\_data\_folder() and get\_results\_folder(). The path returned by get\_data\_folder() has to contain the data i.e. the mandarin_carousel_roi folder from Zenodo. The results will be saved in the path returned by get\_results\_folder(). For example:
```python
from pathlib import Path

def get_data_folder():
    return Path.home() / "scandata"
    
def get_results_folder():
    return Path.home() / "experiments" / "top-ct"
```

To create a conda environment that can run the experiments, follow the instructions in conda environment/create_environment.txt. The exact environment that was used in the paper is also stored in conda environment/environment.yml.
