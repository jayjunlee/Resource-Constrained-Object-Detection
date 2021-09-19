from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time

from annotation import Annotator
import numpy as np
import picamera
from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    #print(input_tensor.shape)
    input_tensor[:, :] = image

def annotate_objects(annotator, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    xmin, ymin, xmax, ymax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)
    print([int(xmin), int(ymin), int(xmax), int(ymax)])
    # Overlay the box, label, and score on the camera preview
    annotator.bounding_box([int(ymin), int(xmin), int(ymax), int(xmax)])
    #print(obj['class_id'])
    annotator.text([int(ymin), int(xmin)],'%s\n' % (labels[obj['class_id']]))

def get_output_tensor(interpreter, index):
  """Return the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)

  print(scores)
  print(boxes)
  print(count)
  print(classes)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {'bounding_box': boxes[i], 'class_id': classes[i], 'score': scores[i]}
      results.append(result)
  return results

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', help='File path of .tflite file.', required=True)
  parser.add_argument('--labels', help='File path of labels file.', required=True)
  parser.add_argument('--threshold', help='Score threshold for detected objects.', required=False, type=float, default=0.4)
  args = parser.parse_args()

  labels = load_labels(args.labels)
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, _, input_height, input_width = interpreter.get_input_details()[0]['shape']

  with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=15) as camera:
    camera.start_preview()
    print("sleep")
    time.sleep(2)
    try:
      stream = io.BytesIO()
      annotator = Annotator(camera)
      while(1):
        camera.capture(stream, format='jpeg')
        stream.seek(0)

        image = Image.open(stream).convert('RGB')
        image = np.asarray(image)
        image = image.astype('uint8')

        start_time = time.monotonic()
        results = detect_objects(interpreter, image, args.threshold)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        annotator.clear()
        annotate_objects(annotator, results, labels)
        annotator.text([5, 0], '%.1fms' % (elapsed_ms))
        annotator.update()

        stream.seek(0)
        stream.truncate()

    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()