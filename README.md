# Resource Constrained CNN

A lightweight convolutional neural network solution to detect obstacle balls using computer vision that is trained using PyTorch / TFLite and implemented on the following resource constrained environments: Intel DE10-Lite FPGA and Raspberry Pi 3.

## Limitations / Constraints
- Limited dataset of about 500 images (approx. 100 for each class)
- Dataset includes images of small objects
- Limited compute power / resources
- Constrained by low target inference time (~= 30 ms)
- Constrained by high target frame rate (30 ~ 60 fps)

## Datasets
Manually taken images are split into three categories in terms of lighting and thus brightness levels: dark, normal and bright. This is to mainly ensure robust inference of NNs under varying light settings. <br>
Each of these images contain a single ball of five different colours: red, green, blue, yellow and pink.

The [dataset](https://github.com/jl7719/FPGA-CNN-Computer-Vision/tree/main/dataset) contains the raw images (1920x1080) and the respective label csv file that contains the dimensions of bounding boxes and the colour of the ball on the image.

![Ball Dataset](balls.png)

## Implementation and Training Model
### Input and output formats
The input tensor to the CNN is RGB image of size 320x240. The output tensors are a tensor of size [1x4] for the object bounding box regression and a tensor of size [1x5] for the classification / scores of each class of the balls.

### Image data augmentation
To make the most out of a limited dataset and to **prevent the model from overfitting** onto the training data, the input images can be augmented i.e. flipped, cropped, etc. just to make sure that the training images are slightly different and the model is not fed in and hence learning the exact same tensors.

### Loss functions
Also known as the cost function, for the dual-inferencing CNN (bbox regression & classification), it is necessary to use the appropriate loss function to be minimised in order to achieve the desirable performance once trained or to even train the NNs. For classification tasks cross entropy loss was used and for bbox regression tasks, L1 loss (Mean Absolute Error) / L2 loss (Mean Squared Error) / IOU loss (Intersection over Union).

### Simple CNN
The initial attempt was to design a CNN architecture with few conv2d layers followed by activations and maxpooling with fc layers in the end. 

~= 30 ms
very low robustness & accuracy
fitted to training set only
x work on images with different backgrounds (high variance)

### Transfer learning pre-trained state-of-the-art CNN
Some state-of-the-art CNNs such as the resnets and mobilenets with pre-trained early layers frozen (great feature extractors) and trainable fully connected layers at the end were trained on the dataset. The training and the progress in validation loss over the number of epochs trained was great compared to a simple CNN (just a few conv2d layers with fc layers). However, when the torch model was converted to TFLite model for deployment on raspberry pi, the **inference time was about > 3000 ms due to the limited compute power**. Although transfer learning is beneficial given a limited, small dataset, the computational cost is too high for a CPU to work in real-time with low inference time.

### Learning rate scheduler

### EfficientDet
EfficientDet-Lite0: ~= 1000 ms
EfficientDet-Lite2: ~= 2700 ms


## Optimizations
### Evaluation Metrics
### Quantization Techniques
- Post-training quantization
- Quantization-aware training
- Torch to TFLite conversion
For pytorch model to TFLite conversion, run the following command on terminal.
```python
python3 torch_to_tflite.py --torch ./trained_model/CNN2.pt --tflite ./model/CNN2.tflite
```
### Pruning

## Room for improvements
### Biased dataset
I started this project by collecting the ball dataset, purely out of my experience from [hard-coded computer vision](https://github.com/rs3319/EE2-Mars-Rover-Project-2021/blob/main/DE10_LITE_D8M_VIP_16/ip/EEE_IMGPROC/EEE_IMGPROC.v) that directly works (only in the optimal light setting) from the individual pixel values that are gaussian filtered to minimise noise. I thought by collecting images of certain categories of light settings (dark, normal, bright), it would help neural networks to generalize better but it turns out that was not the case despite image augmentation and possibly due to [this](https://jmlr.org/papers/volume20/19-519/19-519.pdf) nature of CNNs.
### Importance of neural network accelerators
When deploying several trained models on Raspberry Pi 3, only a minimal CNN e.g. 2 conv2d layers followed by 2 fc layers was just about to satisfy the resource constraints and the target performance in terms of the frame rate and inference time. This leads to how the current processors on edge devices are not solely for DL inferencing and how the software stack or the compilers on top of hardware system adapt the CNNs to optimize for the CPU. The performance on inference can be improved by using AI accelerators such as [Edge TPU](https://coral.ai/products/accelerator/#description) from Google along with the CPU on edge devices.
### Adapting current processors for DL purposes & Limitations
### The need of inference accelerators for neural networking operations

## Related papers
- [A Survey of the Recent Architectures of Deep Convolutional Neural Networks](https://arxiv.org/abs/1901.06032)
- [Transfer Learning](https://cs231n.github.io/transfer-learning/)
- [Pruning](https://jacobgil.github.io/deeplearning/pruning-deep-learning)
- [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
- [ShortcutFusion: From Tensorflow to FPGA-based accelerator with reuse-aware memory allocation for shortcut data](https://arxiv.org/abs/2106.08167)
- [Efficient Methods and Hardware for Deep Learning](https://stacks.stanford.edu/file/druid:qf934gh3708/EFFICIENT%20METHODS%20AND%20HARDWARE%20FOR%20DEEP%20LEARNING-augmented.pdf)

