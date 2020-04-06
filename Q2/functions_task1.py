from utils import *

def compute_edges_dxdy_new(I):
  """Returns the norm of dx and dy as the edge response function."""
  I = I.astype(np.float32)/255.
  # We need a cyclic conditions
  dx = cv2.filter2D(I, cv2.CV_32F, np.array([[-1, 0, 1]]), borderType=cv2.BORDER_REFLECT101)
  dy = cv2.filter2D(I, cv2.CV_32F, np.array([[-1, 0, 1]]).T, borderType=cv2.BORDER_REFLECT101)
  mag = np.sqrt(dx**2 + dy**2)
  mag = mag / np.max(mag)
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
  return mag 

