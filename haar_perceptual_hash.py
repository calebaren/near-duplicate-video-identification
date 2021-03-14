import pywt as pywt
import cv2 as cv2
import numpy as np
import random
import os


def perceptual_hash(
  video_path,
  X,
  U, 
  N = 64,
  hash_length = 32,
  npasses = 4,
  seed = 100):
  # set seed to preserve randomness
  print(video_path)
  random.seed(seed)
  np.random.seed(seed)

  ## read in video
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1

  print(video_path, "successfully loaded video...")

  ## set normalized video dimensions
  new_dim = (X, X)
  
  ## data matrix
  v_norm_trans = np.zeros(tuple(list(new_dim)+ [frame_count]))

  i = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      if i == 1:
        print("reading video...")
      # grayscale and normalized
      gray_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), new_dim)
      v_norm_trans[:, :, i] = gray_resized
        
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    else:
      print("ERROR: Check file path and/or video file. Unsuccessfully read.")
      return None
      break
    i += 1
  print("video resized successfully to size", v_norm_trans.shape)
  cap.release()

  ## interpolate
  new_frame_count = 2 ** int(np.log2(v_norm_trans.shape[2])+1)

  ## add new_frame_count - video_array.shape[2] copies
  inter_frame = np.mean(v_norm_trans, axis = 2)

  interpolated_video = np.concatenate([v_norm_trans,
                                      np.repeat(inter_frame[:, :, np.newaxis], 
                                                new_frame_count - v_norm_trans.shape[2], 
                                                axis = 2)], 
                                      axis = 2)

  # cv2_imshow(np.mean(v_norm_trans, axis = 2))
  cv2.destroyAllWindows()
  v_norm_trans = interpolated_video

  ## apply haar wavelet 4 times
  data = v_norm_trans
  approximations, details = [], []
  for i in range(npasses):
    cA, cD = pywt.dwt(data, 'haar')
    approximations.append(cA)
    details.append(cD)
    data = cA

  print("wavelet transformation successful...")
  ## final pass assigns low pass to data

  ## select N samples of U x U x U samples
  X, _, T = data.shape
  Ux = random.choices(range(X - U), k=N)
  Uy = random.choices(range(X - U), k=N)
  Ut = random.choices(range(T - U), k=N)

  ## feature matrix
  F = []
  for x, y, t in zip(Ux, Uy, Ut):
    F.append(np.ravel(data[x:(x+U), y:(y+U), t:(t+U)]))
  F_ = np.array(F).T

  d,_ = F_.shape
  k = hash_length
  
  ## random projection matrix: R_arm = (k, D) matrix
  R_arm = np.random.choice([-0.5, 0.5], size=(k, d), replace=True)

  ## random projection: H = (k, N) matrix
  H = 1/np.sqrt(k) * R_arm @ F_

  ## hash computation: h' = (k) - vector
  h_prime = np.mean(H, axis = 1)

  # final hash: binarize
  return np.int8(h_prime >= np.median(h_prime))

# usage:
test_video_path = 'test_video.flv'
hash = hashing(video_path = test_video_path, X = 60, U = 32)