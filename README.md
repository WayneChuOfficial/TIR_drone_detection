# TIR_drone_detection
Thermal Infrared Drone Detection

## Usage
### Data preparation
Unzip the UAV dataset to the dir "dataset" and rename it as "UAV_dataset" like
```
|-- dataset
    |-- UAV_dataset
        |--train
           |--...
        |--test
           |--...
```
Then run the python file "preparedataset.py" to split the train and val dataset, and get the correspinding mask of each image.
```
python preparedataset.py
```
