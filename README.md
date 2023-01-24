# Beat perception
___
___
### This is a repository for beat perception project!!


## Dependencies
Python 3, PyTorch(1.9.0)

Other dependencies can be installed via 

  ```pip install -r requirements.txt```

___

## Run Demos

### Train
* To run the training of experiment ```X``` where ```X``` is a config file for a session :

  ```python run_exp.py -c config/X.yaml```

### Test

* After training, you can specify the ```test_model``` field of the configuration yaml file with the path of your best model snapshot, e.g.,

  ```test_model: exp/Liangwei/Liangwei_hcp/model_snapshot_best.pth```	

* To run the test of experiments ```X```:

  ```python run_exp.py -c config/X.yaml -t```
