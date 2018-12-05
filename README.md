# Transfer Learning with Inception-v3 for Histopathological Image Classification

The classify.py, convert-model.py, label.py, predict-request.py, retrain.py files were taken from the TensorFLow transfer learning api -- https://www.tensorflow.org/hub/tutorials/image_retraining.

The BreakHis dataset was used for the image classification, and available here -- https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/. Download and move dataset to this repository.

## File summaries
#### '1. Dataset_Maker.ipynb': 
The BreakHis dataset contains histopathology images in 40x, 100x, 200x, 400x  magnifications. This file creates seperate directories for each magnification in the tf_files directory. This is done because an image classifier will be created using the 40x, 100x, 200x, and 400x datasets. The accuracies of each of the 4 classifiers is compared.

#### '2. Transfer_Learning.ipynb': 
Transfer learning is done on the last layer of Inception-v3, a pre-trained convolutional neural netowrk.

#### '3. Data_Analysis.ipynb': 
The accuracies of the classifiers are compared.
