# AIvengers
A BME Mélytanulás tárgyához kapcsolódó projekt feladat git oldala.

### Csapattagok:
 - Frecska Hajnalka (C1MTMR)
 - Hugauf Dániel Bálint (F3G8I9)
 - Wágner Bánk (ANKNFJ)

### Project description:

In this project, you'll dive into the idea of using multiple models together, known as model ensembles, to make our deep-learning solutions more accurate. They are a reliable approach to improve the accuracy of a deep learning solution for the added cost of running multiple networks. Using ensembles is a trick that's widely used by the winners of AI competitions. The task of the students: explore approaches to model ensemble construction for semantic segmentation, select a dataset (preferentially cardiac MRI segmentation, but others also allowed), find an open-source segmentation solution as a baseline for the selected dataset, and test it. Train multiple models and construct an ensemble from them. Analyze the improvements, benefits, and added costs of using an ensemble.
# Milestone-3:
#### Running the Pipeline:
 1. enter the folder docker_network
 2. in the command line give the command: **docker-compose up**
 3. it should create 2 images and the pipeline currently runs the test dataset and gives back the evaluation for that into a different file

#### Running the Ensemble:
 1. enter the folder milestone 3
 2. the **create_ensemble.ipynb** file contains the pipeline of how to create the ensemble models
 3. to run the ensemble simply follow the instructions within the notebooks
#### Other Files:
 5. the **models_train.py** contains the training for each model, train_models.ipynb contains the training process
 6. **prep.py** will preprocess the images and **ensemble.py** will create the ensemble models

# Milestone-2:

### How-to Run:

#### Running the Pipeline:
 1. enter the folder docker_network
 2. in the command line give the command: **docker-compose up**
 3. it should create 2 images and the pipeline currently runs the test dataset and gives back the evaluation for that into a different file

#### Running the Training:
 1. enter the folder milestone 2
 2. the **X_model.ipynb** file contains the 4 models used for our ensemble
 3. the **X_files.zip** contains generated test results and the additional plots
 4. to run each model simply follow the instructions within the notebooks



# Milestone-1:

### How to Run:
**base_model.ipynb**
 1. if you can't find any of the following, you can download it from this link: https://drive.google.com/drive/folders/1EP5HSA__aVHqcikNAYlxRSWfL89ijlyR?usp=share_link
 2. upload **base_model.ipynb** to Colab
 3. after running the installs, follow the instructions provided in the 2nd cell (delete ".generic_utils" from the appropriate file, save it, and restart the runtime)
 4. download the **ACDC.zip** file (according to the 3rd cell), then unzip it (4th cell)
 5. run the cells, and keep paying attention to the comments for further guidance
    
**prep.ipynb**
 1. upload **prep.ipynb** to colab
 2. running the cells, it will download the dataset and extract files.
 3. after the all cells runned, you will get all .npy files. These contain the preprocessed and augmented images and gt-s in numpy.array format
 4. We will use the files for teaching later

### File descriptions:
  1. **base_model.ipynb** -  contains the foundational model, providing a benchmark against which the subsequent results of the ensemble can be compared
  2. **ACDC.zip** - contains the dataset, with 600 training images, 300 testing images, and the corresponding labels
  3. **\*.png** - the plots generated by the notebook, intended for visual reference
  4. **docker_network** folder contains the base files for the docker. Each folder contains a requirements.txt a python file and a Dockerfile for the image. In the main folder, there is the docker-compose file, which runs the containers after the images are built.
  5. **prep.ipynb** - read the ACDC dataset, resize and normalize the images, make data augmentations, and save the numpy arrays to .npy files

### Related Works:
 - https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb
 - https://github.com/vlbthambawita/divergent-nets
 - https://github.com/WalBouss/SenFormer
 - https://arxiv.org/abs/2107.00283
 - https://arxiv.org/abs/2111.13280
 - https://dl.acm.org/doi/abs/10.1145/3555776.3577682 (no official code)
 - https://www.creatis.insa-lyon.fr/Challenge/acdc/
 - https://youtu.be/NFIYdYjJams?si=thuXzUgsZ4XEdxCT

