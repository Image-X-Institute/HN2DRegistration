# HN2DRegistration
Code for training and testing a 2D-2D deformable registration network, designed for registering 2D kV x-ray images to synthetic x-ray images derived from patient planning data. 

**Author:** Mark Gardner

Code for training and testing a 2D-2D deformable registration network, designed for registering 2D kV x-ray images to synthetic x-ray images derived from patient planning data, 
as featured in the paper **insert proper citation**.
Code is based off of voxelmorph approach (https://github.com/voxelmorph/voxelmorph), as well as voxelmap (https://github.com/Image-X-Institute/Voxelmap).

For any questions please email mark.gardner@sydney.edu.au

## Setup/Build/Install


Clone repository from github: 
```
git clone https://github.com/Image-X-Institute/HN2DRegistration
```

Python version required is <=3.10. Setup python environment using:
```
pip install -r requirements.txt
```

Make sure pytorch is installed using the appropriate conda version (https://pytorch.org/get-started/previous-versions/) if you want GPU enhanced training.


## Usage

Test moving and fixed images are located in the [examples folder](https://github.com/Image-X-Institute/HN2DRegistration/tree/master/examples)

The network can be trained using the command line command:

```
python train.py --name RegTest --movingDir <path to moving images> --fixedDir <path to fixed images> --lr 0.0001 --niter 20
```

and tested usign the command line command:

```
python test.py --name RegTest --movingDir <path to test moving images> --fixedDir <path to test fixed images> --maskDir <path to test moving image masks> --epoch <EpochNumber>
```

You can train using cpu only by specifying ```--gpu_ids -1``` as an option for the testing and training code. 

Running the train.py code will create a directory in <checkpoints_dir> which create a loss plot, a log file and for every epoch will create:
- A copy of the trained model.
- An example of the moving, fixed and moved (deformed moving) image.

Runnine the test.py code will create a directory in <results_dir> which will output the fixed images, moving images, the deformed mask and OAR images if defined, and images showing the 2D DVF outputs overlayed on the moving images. 

More information on the train and test command inputs can be found using 

```
python train.py -h
python test.py -h
```

## Datasets
The training dataset was from the cancer imaging archive [HNSCC dataset](https://www.cancerimagingarchive.net/collection/hnscc/). The testing set was from a 30 patient dataset from Blacktown Hospital, Sydney Australia and 
can be downloaded from the University of Sydney [library](https://hdl.handle.net/2123/34250). 

## Citation

If using this work please cite as **insert proper citation**.

## Directory Structure

- [Examples](https://github.com/Image-X-Institute/HN2DRegistration/tree/master/examples): Example data
- [Supporting](https://github.com/Image-X-Institute/HN2DRegistration/tree/master/supporting): Supporting code/functions including code for the network model and the dataset.  
