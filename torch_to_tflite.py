import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=17,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=30,kernel_size=13,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=90,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.dense = nn.Sequential(
            nn.Linear(540,128,bias=True),
            nn.ReLU()
        )
        self.colour = nn.Linear(128,5,bias=True)
        self.bbox = nn.Linear(128,4,bias=True)
    def forward(self,x):
        output_conv = self.conv3(self.conv2(self.conv1(x)))
        output_flat = output_conv.view(output_conv.shape[0],-1)
        output_dense = self.dense(output_flat)
        output_colour = self.colour(output_dense)
        output_bbox = self.bbox(output_dense)
        return output_bbox, output_colour

def torch2tflite(torch_path, tflite_path):
    # Sample input image
    sample_input = "./dataset/raw/image_blue_51.jpg"
    sample_input = cv2.resize(cv2.imread(sample_input),(240,320))
    x = cv2.cvtColor(sample_input.astype(np.float32), cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(np.rollaxis(x, 2))[None,]
    example_input = img

    print("Sample input found.")

    # Torch object class to load the trained model
    pytorch_model = CNN()
    pytorch_model.load_state_dict(torch.load(torch_path,map_location=torch.device('cpu')))

    # Intermediate ONNX and TF conversion file paths
    name = torch_path[0:-3]
    ONNX_PATH="./model/" + name + ".onnx"
    TF_PATH = "./model/" + name + "tf"

    torch.onnx.export(model=pytorch_model, args=example_input, f=ONNX_PATH, verbose=False, export_params=True, do_constant_folding=False, input_names=['input'], output_names=['colour','bbox'])
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_PATH)

    print("ONNX conversion complete.")

    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    open(tflite_path, 'wb').write(tf_lite_model)

    print("TFLite conversion complete.")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--torch', help='File path of torch model (.pt) input file.', required=True)
    parser.add_argument('--tflite', help='File path of tflite output file.', required=True)
    args = parser.parse_args()
    TORCH_PATH = args.torch
    TFLITE_PATH = args.tflite
    if not os.path.exists('./model'):
        os.mkdir('./model')
    torch2tflite(TORCH_PATH,TFLITE_PATH)

if __name__ == '__main__':
    main()
