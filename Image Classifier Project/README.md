# Project Image Classifier Project
**Summary:** In this project, I train an image classifier to recognize different species of flowers using PyTorch. The Jupyter notebook builds and trains a deep neural network on a flower data set (data not included here), which is then converted into an application that others can use via the predict.py and train.py scripts.

The Jupyter notebook is broken down into the following main steps:
1. Load and preprocess the image dataset
2. Train the image classifier on the dataset
3. Use the trained classifier to predict image content

## Application
* train.py script will train a new network on a dataset and save the model as a checkpoint. 
* predict.py uses a trained network to predict the class for an input image.

**Script Details and Usage**

train.py trains a new network on a data set and prints out training loss, validation loss, and validation accuracy as the network trains.  

The training script also allows users to choose from two different architectures available from torchvision.models (vgg16 and densenet121), allows users to set hyperparameters for learning rate, number of hidden units, and training epochs, as well as lets users choose to train the model on a GPU.

**Basic usage:** 
```
python train.py data_directory
```
* Options:
    - Set directory to save checkpoints: 
			```
			python train.py data_dir --save_dir save_directory
			```
			
   - Choose architecture: 
		     ```
			python train.py data_dir --arch "vgg16"
		   ```
		   
    - Set hyperparameters: 
		    ```
			python train.py data_dir --lr 0.01 --hidden_units 512 --epochs 20
			 ```
			 
    - Use GPU for training:
		    ```
	    python train.py data_dir --gpu
			```
---

predict.py predicts the flower name from an image along with the probability of that name.
 
Essentially, the script reads in an image and a checkpoint, then prints the most likely image class and it's associated probability.  It allows users to print out the top K classes along with associated probabilities, allows users to load a JSON file that maps the class values to other category names, as well as allows users to use the GPU to calculate the predictions.

**Basic usage:** 
```
python predict.py /path/to/image checkpoint
```
* Options:
    - Return top K most likely classes: 
			 ```
			python predict.py image_path checkpoint --topk 3
			```
			
    - Use a mapping of categories to real names: 
			```
			python predict.py image_path checkpoint --category_names cat_to_name.json
			```
    - Use GPU for inference: 
			```
			python predict.py image_path checkpoint --gpu
			```