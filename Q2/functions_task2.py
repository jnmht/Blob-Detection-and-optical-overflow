from utils import *

def gkern(ksize, sigma):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(ksize, std=sigma).reshape(ksize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def apply_derivative_gaussian_filter_(gray, sigma):
    kernel = gkern(sigma*2, sigma).astype(np.float32)
    kernel_dx = cv2.filter2D(kernel, cv2.CV_32F, np.array([[-1, 0, 1]]), borderType=cv2.BORDER_REFLECT101)
    kernel_dy = cv2.filter2D(kernel, cv2.CV_32F, np.array([[-1, 0, 1]]).T, borderType=cv2.BORDER_REFLECT101)
    dx = cv2.filter2D(gray, cv2.CV_32F, kernel_dx, borderType=cv2.BORDER_REFLECT101)
    dy = cv2.filter2D(gray, cv2.CV_32F, kernel_dy, borderType=cv2.BORDER_REFLECT101)
    mag = np.sqrt(dx**2 + dy**2)
    return dx, dy, mag

def apply_derivative_gaussian_filter(sigma, gray):
    dx, dy, mag = apply_derivative_gaussian_filter_(gray, sigma)
    mag = np.sqrt(dx**2 + dy**2)
    mag = mag / np.max(mag)
    mag = mag * 255.
    mag = np.clip(mag, 0, 255)
    mag = mag.astype(np.uint8)
    return mag
