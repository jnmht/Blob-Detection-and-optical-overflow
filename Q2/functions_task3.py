from utils import *

from functions_task2 import *

def dist_miltiplier(dx,distance,x):
    final=x + dx*distance
    return final

def get_neighbours(I, dist, dx_norm, dy_norm):
    x,y = np.meshgrid(np.arange(0, I.shape[1]), np.arange(0, I.shape[0]))
    x_ = dist_miltiplier(dx_norm,dist,x)
    y_ = dist_miltiplier(dy_norm,dist,y)
    #r = my_remap(I, x_, y_)
    r = cv2.remap(I.astype(np.float32), x_.astype(np.float32), y_.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return r

def getnorm(dx,norm):
    return dx/(norm + 1e-6)

def x_copy(d_norm,x):
    g=x + d_norm*1
    return g

def apply_nonmax_supression(mag, dx, dy):
    norm = np.sqrt(dx*dx + dy*dy)
    dx_norm = getnorm(dx,norm)
    dy_norm = getnorm(dy,norm)
    x,y = np.meshgrid(np.arange(0, mag.shape[1]), np.arange(0, mag.shape[0]))
    x_ = x_copy(dx_norm,x)
    y_ = x_copy(dy_norm,y)
    M = mag.copy()

    for d in [-1, 1]:
        next_mag = get_neighbours(mag, d, dx_norm, dy_norm)
        M[M <= next_mag] = 0
    return M  

def apply_derivative_gaussian_nms(sigma, image):
    dx, dy, mag = apply_derivative_gaussian_filter_(image, sigma)
    mag = np.sqrt(dx**2 + dy**2)
    mag = apply_nonmax_supression(mag, dx, dy)
    mag = apply_nonmax_supression(mag, dx, dy)
    mag = mag / np.max(mag)
    mag = mag * 255.
    mag = np.clip(mag, 0, 255)
    mag = mag.astype(np.uint8)
    return mag  
