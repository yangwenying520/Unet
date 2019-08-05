import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
import time
import torch


def dense_crf(img, output_probs):
    """
    input:
      img: numpy array of shape (height, width, num of channels) format:RGB
      prob: numpy array of shape (n_classes, height, width), neural network last layer sigmoid output for img
            output_probs is the result of log_sofmax
    output:
      res: (n_classes, height, width)
    """
    img = img[:, :, ::-1]#opencv读取的图片为BGR格式，需要转换
    n_classes = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]
    d = dcrf.DenseCRF2D(w, h, n_classes)
    U = -(output_probs)#U is negative log logarithm
    U = U.reshape((n_classes, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=10, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img, compat=10)

    Q = d.inference(5)
    # print(np.array(Q).shape)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return (torch.from_numpy(Q)).view(-1,h,w)#转换成(batches,h,w)
