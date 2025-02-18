import cv2
import numpy as np

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    
    cam = np.uint8(255*cam)
    cam = cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
    return cam

def normalize(m):
    m = (m-m.min()) / (m.max()-m.min())
    return m.permute(0,2,3,1).detach().cpu().numpy()

def normalize_last_dim(m):
    min_v = m.min(dim=-1, keepdim=True)[0]
    return (m-min_v) / (m.max(dim=-1,keepdim=True)[0]-min_v)
    