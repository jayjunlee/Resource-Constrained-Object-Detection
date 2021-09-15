# Resource Constrained CNN

A lightweight convolutional neural network solution to detect obstacle balls using computer vision that is trained using PyTorch / TFLite and implemented on the following resource constrained environments: Intel DE10-Lite FPGA and Raspberry Pi 3.

## Limitations / Constraints
- Limited dataset of about 500 images (approx. 100 for each class)
- Limited compute power / resources
- Constrained by low target inference time (~= 30 ms)
- Constrained by high target frame rate (30 ~ 60 fps)

## Datasets
Manually taken images are split into three categories in terms of lighting and thus brightness levels: dark, normal and bright. And these can be easily distinguished by the image file names. <br>
Each of these images contain a single ball of five different colours: red, green, blue, yellow and pink.

The [dataset](https://github.com/jl7719/FPGA-CNN-Computer-Vision/tree/main/dataset) contains the raw images (1920x1080) and the respective label csv file that contains the dimensions of bounding boxes and the colour of the ball on the image.

## Implementation and Training Model
### Input and output formats
The input tensor to the CNN is RGB image of size 320x240. The output tensors are a tensor of size [1x4] for the object bounding box regression and a tensor of size [1x5] for the classification / scores of each class of the balls.

### Image data augmentation
To make the most out of a limited dataset and to **prevent the model from overfitting** onto the training data, the input images can be augmented i.e. flipped, cropped, etc. just to make sure that the training images are slightly different and the model is not fed in and hence learning the exact same tensors.

### Simple CNN
The initial attempt was to design a CNN architecture with few conv2d layers followed by activations and maxpooling with fc layers in the end.

### Transfer learning pre-trained state-of-the-arts CNN
Some state of the arts CNNs such as the resnets and mobilenets with pre-trained early layers frozen (great feature extractors) and trainable fully connected layers at the end were trained on the dataset. The training and the progress in validation loss over the number of epochs trained was great compared to a simple CNN (just a few conv2d layers with fc layers). However, when the torch model was converted to TFLite model for deployment on raspberry pi, the **inference time was about > 3000 ms due to the limited compute power**. Although transfer learning is beneficial given a limited, small dataset, the computational cost is too high for a CPU to work in real-time with low inference time.



The current implementation of the training of the CNN is in [here](https://github.com/jl7719/FPGA-CNN-Computer-Vision/blob/main/cnn_ball_detection.ipynb). With the limited dataset and compute power, the raw images are converted to 320x240 sized images which overall requires less trainable parameters in the nn and is also useful where training can easily end up overfitting the datasets. To avoid overfitting and minimising the validation loss, images are slightly [augmented](https://github.com/jl7719/FPGA-CNN-Computer-Vision/blob/main/data_augmentation.py) before it is used for training. The trained model is stored as .pt [file](https://github.com/jl7719/FPGA-CNN-Computer-Vision/tree/main/trained_model) which can be loaded for use.
