import numpy as np
import cv2
from itertools import product
from math import ceil


def priorbox(min_sizes, steps, clip, image_size):
    feature_maps = [[ceil(image_size[0] / step),
                     ceil(image_size[1] / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        t_min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in t_min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = np.array(anchors, dtype=np.float32).reshape((-1, 4))
    if clip:
        output = np.clip(output, 0, 1)
    return output

def decode(loc, priors, variances):
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ), axis=1)
    
    boxes[:, :2] -= boxes[:, 2:] * 0.5
    # boxes[:, 2:] += boxes[:, :2]
    return boxes  ###（x, y, w, h）


def decode_landm(pre, priors, variances):
    return np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ), axis=1)


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[0]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp, [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # _, k = cv2.solve(A.T@A, A.T@b, cv2.DECOMP_CHOLESKY)   ////等价的， c++的solve(A.t()*A, A.t()*b, k, DECOMP_CHOLESKY);

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    h0, w0 = img.shape[:2]
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = int(left + target_size)
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = int(up + target_size)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    img = img[up:below, left:right]
    dsth, dstw = int(target_size), int(target_size)
    if img.shape[0] < dsth:
        img = cv2.copyMakeBorder(img, 0, dsth - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    if img.shape[1] < dstw:
        img = cv2.copyMakeBorder(img, 0, 0, 0, dstw - img.shape[1], cv2.BORDER_CONSTANT, value=0)

    if mask is not None:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = mask[up:below, left:right]
        if mask.shape[0] < dsth:
            mask = cv2.copyMakeBorder(mask, 0, dsth - mask.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
        if mask.shape[1] < dstw:
            mask = cv2.copyMakeBorder(mask, 0, 0, 0, dstw - mask.shape[1], cv2.BORDER_CONSTANT, value=0)

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --numpy.array  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --numpy.array  (target_size, target_size)
    
    Parameters:
        img                --numpy.array  (raw_H, raw_W, 3)
        lm                 --numpy.array  (5, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --numpy.array  (raw_H, raw_W, 3)
    """

    h0, w0  = img.shape[:2]
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p, lm3D)
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    # trans_params = np.array([w0, h0, s, t[0], t[1]])
    trans_params = np.array([w0, h0, s, t[0][0], t[1][0]])

    return trans_params, img_new, lm_new, mask_new

def process_uv(uv_coords, uv_h = 224, uv_w = 224):
    uv_coords[:,0] = uv_coords[:,0] * (uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1] * (uv_h - 1)
    # uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords
