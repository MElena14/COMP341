import numpy as np
from numpy import cos, sin

from IPython import display
import PIL
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import torch
import cv2

from rect_metric import RotatedRect, compare_grasps

def draw_box(image,center_point, rotate, width, height, color):
    angle = np.radians(rotate)
    # Determine the coordinates of the 4 corner points
    rotated_rect_points = []
    x = center_point[0] + ((width / 2) * cos(angle)) - ((height / 2) * sin(angle))
    y = center_point[1] + ((width / 2) * sin(angle)) + ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])
    x = center_point[0] - ((width / 2) * cos(angle)) - ((height / 2) * sin(angle))
    y = center_point[1] - ((width / 2) * sin(angle)) + ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])
    x = center_point[0] - ((width / 2) * cos(angle)) + ((height / 2) * sin(angle))
    y = center_point[1]- ((width / 2) * sin(angle)) - ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])
    x = center_point[0] + ((width / 2) * cos(angle)) + ((height / 2) * sin(angle))
    y = center_point[1] + ((width / 2) * sin(angle)) - ((height / 2) * cos(angle))
    rotated_rect_points.append([x,y])

    rotatedImg = cv2.polylines(image, np.array([rotated_rect_points], np.int32), True, color)
    return rotatedImg


def rotated_rectangle(image, predicted, correct):
    if torch.is_tensor(image):
      image = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
      image = image.permute(1, 2, 0).numpy().astype('uint8')
      image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)

    color = (0, 255, 0)
    x, y, angle, width, height = predicted
    predicted_result = draw_box(image, (x,y), angle, width, height, color)

    color = (0, 0, 255)
    x, y, angle, width, height = correct
    final = draw_box(predicted_result, (x,y), angle, width, height, color)
    print('score =', compare_grasps(predicted, correct))
    cv2_imshow(final)

"""A replacement for cv2.imshow() for use in Jupyter notebooks.

  Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
"""
def cv2_imshow(a):
  if torch.is_tensor(a) or isinstance(a, np.ndarray):
    if torch.is_tensor(a):
        a = a.cpu().numpy()
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
      if a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
      else:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(a)
  else:
    img = a
  display.display(img)

# functions to show an image
def imshow(img):
    """
    :param img: (PyTorch Tensor)
    """
    if torch.is_tensor(img):
      # unnormalize
      img = img / 2 + 0.5     
      # Convert tensor to numpy array
      img = img.numpy()
      # Color channel first -> color channel last
      img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.grid(False)
    plt.axis('off')

def visul_grasps(rect1, rect2):
    r1 = RotatedRect(*rect1)
    r2 = RotatedRect(*rect2)
    print("score: ", compare_grasps(rect1, rect2))
    
    fig = plt.figure(1, figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.set_xlim(500, 800)
    ax.set_ylim(400, 700)
    ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
    ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
    ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))
    plt.show()

class UnNormalize(object):
  def __init__(self, mean, std):
      self.mean = mean
      self.std = std

  def __call__(self, tensor):
      for t, m, s in zip(tensor, self.mean, self.std):
          t.mul_(s).add_(m)
      return tensor * 256