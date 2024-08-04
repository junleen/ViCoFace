import numpy as np
import torch
import cv2
import kornia
import torch.nn.functional as F


def bbox2point(left, right, top, bottom, type='bbox', vshift_scale=0.12):
    ''' bbox from detector and landmarks are different
    vshift_scale = 0.12 for face recognition, 0.06 for 3d coefficient estimation
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 - vshift_scale*old_size], dtype=np.int32)
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.16], dtype=np.int32)
    elif type=='kpt5':
        old_size = (right - left + bottom - top)/2*1.25
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*vshift_scale], dtype=np.int32)
    else:
        raise NotImplementedError
    return old_size, center
        
def kpt_estimate_bbox(kpt, vshift_scale=0.12):
    '''Return bbox with given 68 key points'''
    if len(kpt) == 68:
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0])
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        old_size, center = bbox2point(left, right, top, bottom, type='kpt68', vshift_scale=vshift_scale)
    elif len(kpt) == 4:
        x, y, w, h = kpt
        old_size, center = bbox2point(x, x+w, y, y+h, type='bbox', vshift_scale=vshift_scale)
    elif len(kpt) == 5:
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0])
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        old_size, center = bbox2point(left, right, top, bottom, type='kpt5', vshift_scale=vshift_scale)
    size = old_size * 1.25//2*2
    bbox = np.array([center[0]-size/2, center[1]-size/2, size, size])
    return bbox

def torch_kpt_array2bbox(lmks, vshift_scale=0.12):
    bboxes = []
    lmks = lmks.detach().cpu().numpy()
    for lmk in lmks:
        bbox = kpt_estimate_bbox(lmk, vshift_scale)
        bboxes.append(bbox)
    bboxes = torch.from_numpy(np.vstack(bboxes))
    return bboxes

def bbox2AffineMatrix(bbox, size):
    '''
    input:
        bbox: Tensor (4, )
    return:
        Tensor 2 x 3 # affine tranform matrix
    '''
    x, y, w, h = bbox[:4]
    DST_PTS = np.array([[0,0], [0, size-1], [size-1, 0]], dtype=np.float32)
    src_pts = np.array([[x, y], [x, y+h], [x+w, y]], dtype=np.float32)
    M = torch.Tensor(cv2.getAffineTransform(src_pts, DST_PTS)).float()
    return M

def estimate_single_transform_torch(lm_68p, size=224, vshift_scale=0.12):
    '''
    input:
        lm_68p: Tensor 68 x 2
    return:
        Tensor 2 x 3 # affine tranform matrix
    '''
    x, y, w, h = kpt_estimate_bbox(lm_68p.cpu().numpy(), vshift_scale=vshift_scale) # np.ndarray
    M = bbox2AffineMatrix([x, y, w, h], size)
    M = M.to(lm_68p.device)
    return M

def estimate_transform_torch(points, size=224, vshift_scale=0.12):
    '''
    input:
        points: Tensor B x 68 x 2 or B x 4 
    return:
        Tensor B x 2 x 3 # affine tranform matrix
    '''
    M = []
    if max(points.shape[1:]) > 4:
        lm_68p_ = points.detach().cpu().numpy()
        for i in range(lm_68p_.shape[0]):
            x, y, w, h = kpt_estimate_bbox(lm_68p_[i], vshift_scale=vshift_scale) # np.ndarray
            M.append(bbox2AffineMatrix([x, y, w, h], size))
    else:
        bbox = points.detach().cpu().numpy()
        for i in range(bbox.shape[0]):
            M.append(bbox2AffineMatrix(bbox[i], size))
    M = torch.stack(M, dim=0).to(points.device)
    return M

def torchKP68CropWarpAffine(image, lm_68p, crop_size=224, hard_crop=False, vshift_scale=0.12, align_corners=True):
    # image: (B, C, H, W) Tensor
    # lm_68p: (B, 68, 2), [x, y] Tensor
    # vshift_scale = 0.12 for face recognition, 0.06 for 3d coefficient estimation
    if hard_crop:
        dsts = []
        lm_68p_ = lm_68p.detach().cpu().numpy()
        for i in range(len(lm_68p_)):
            crop_box = kpt_estimate_bbox(lm_68p_[i], vshift_scale=vshift_scale)
            x, y, w, h = crop_box.astype(np.int32)
            dsts.append(F.interpolate(image[i:i+1, :, y:y+h, x:x+w], (crop_size, crop_size), mode='bilinear', align_corners=align_corners))
        dst_image = torch.cat(dsts, dim=0)
    else:
        M = estimate_transform_torch(lm_68p, size=crop_size, vshift_scale=vshift_scale)
        dst_image = kornia.geometry.warp_affine(image, M, dsize=(crop_size, crop_size), mode='bilinear')
    return dst_image

def torchWarpAffine(image, M, crop_size=224):
    # image: (B, C, H, W) Tensor
    # M: (B, 2, 3), Tensor
    # vshift_scale = 0.12 for face recognition, 0.06 for 3d coefficient estimation
    # dst_image = kornia.geometry.warp_affine(image, M, dsize=(crop_size, crop_size), mode='bilinear')
    dst_image = kornia.geometry.warp_affine(image, M, dsize=(crop_size, crop_size))
    return dst_image

def cv2CropWarpAffine(image, crop_box, crop_size=224, interpolate=cv2.INTER_LINEAR):
    # crop an image by given crop_box, and resize to (crop_size, crop_size)
    DST_PTS = np.array([[0,0], [0, crop_size-1], [crop_size-1, 0]], dtype=np.float32)
    x, y, w, h = crop_box if isinstance(crop_box, np.ndarray) or isinstance(crop_box, tuple) or isinstance(crop_box, list) else crop_box.cpu().numpy()
    src_pts = np.array([[x, y], [x, y+h-1], [x+w-1, y]], dtype=np.float32)
    tform = cv2.getAffineTransform(src_pts, DST_PTS)
    dst_image = cv2.warpAffine(image, tform, dsize=(crop_size, crop_size), flags=interpolate)
    return dst_image

