import torch
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as T
from torchvision import models

preprocess = T.Compose([
                        T.Resize(256),
                        #T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

def decode_segmap(image, nc=21):
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  label_colors = np.array([(0, 0, 0)]*15 + [(255, 255, 255)] + [(0, 0, 0)]*5)

  r = np.zeros_like(image).astype(np.uint8)

  g = np.zeros_like(image).astype(np.uint8)

  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb

import cv2

def make_me_invisible(fore_img, back_img):
  inp = preprocess(fore_img).unsqueeze(0)
  fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
  out = fcn(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  masked = decode_segmap(om)

  masked = Image.fromarray(masked)
  masked = masked.filter(ImageFilter.MaxFilter(size = 19))

  masked = np.asarray(masked)  

  fore_img = np.asarray(fore_img)
  fore_img = cv2.resize(fore_img, (masked.shape[1], masked.shape[0]))

  back_img = cv2.resize(back_img, (masked.shape[1], masked.shape[0]))
  foreground = cv2.bitwise_and(back_img, masked)
  foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

  masked = 255 - masked
  background = cv2.bitwise_and(fore_img, masked)
  background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

  merged = cv2.add(foreground, background)
  return merged

fore_img = cv2.imread("foreground.jfif")
back_img = cv2.imread("background.jfif")

fore_img = Image.fromarray(fore_img)

output = make_me_invisible(fore_img, back_img)
plt.imshow(output)
plt.show()

