# FPGA CNN Ball Detector

A lightweight convolutional neural network solution to detect balls / obstacles using computer vision that is trained using PyTorch and implemented on a resource constrained FPGA with a camera that perceives the surrounding environment.

## Datasets
Manually taken images are split into three categories in terms of lighting and thus brightness levels: dark, normal and bright. And these can be easily distinguished by the image file names. <br>
Each of these images contain a single ball of five different colours: red, green, blue, yellow and pink.

The [dataset](https://github.com/jl7719/FPGA-CNN-Computer-Vision/tree/main/dataset) contains the raw images (1920x1080) and the respective label csv file that contains the dimensions of bounding boxes and the colour of the ball on the image.

## Implementation and Training Model
The current implementation of the training of the CNN is in [here](https://github.com/jl7719/FPGA-CNN-Computer-Vision/blob/main/cnn_ball_detection.ipynb). With the limited dataset and compute power, the raw images are converted to 320x240 sized images which overall requires less trainable parameters in the nn and is also useful where training can easily end up overfitting the datasets. To avoid overfitting and minimising the validation loss, images are slightly [augmented](https://github.com/jl7719/FPGA-CNN-Computer-Vision/blob/main/data_augmentation.py) before it is used for training. The trained model is stored as .pt [file](https://github.com/jl7719/FPGA-CNN-Computer-Vision/tree/main/trained_model) which can be loaded for use.
