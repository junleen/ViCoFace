import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from augmentations import AugmentationTransform
from PIL import ImageFile
from utils import warp_func

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Vox256(Dataset):
    def __init__(self, split, size=256, transform=None, augmentation=False):
        
        self.split = split
        if split == 'train':
            self.ds_path = '../../dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/train'
            self.is_train = True
        elif split == 'test':
            self.ds_path = '../../dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/test'
            self.is_train = False
        else:
            raise NotImplementedError
            
        assert os.path.exists(self.ds_path)

        self.videos = sorted(glob.glob(os.path.join(self.ds_path, 'id*'))) # os.listdir(self.ds_path)
        # videos_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_videos.pt')
        # if os.path.exists(videos_path):
        #     print('load pre-defined video names')
        #     self.videos = torch.load(videos_path)
        # else:
        #     self.videos = sorted(glob.glob(os.path.join(self.ds_path, 'id*'))) # os.listdir(self.ds_path)
        #     torch.save(self.videos, videos_path)

        self.augmentation = augmentation
        self.aug = AugmentationTransform(False, False, True)

        self.transform = transform
        self.index_list = list(range(5000))
        
        self.lmk_scale = size / 320
        self.preload_deca_bbox()
        self.use_bbox = False
        if self.use_bbox:
            self.preload_arcface_bbox()
        else:
            self.preload_arcface_M()

        self.vshift_scale_for_arcface = 0.12
        self.img_size_for_arcface = 112

        self.vshift_scale_for_deca = 0.06
        self.img_size_for_deca = 224

        self.dataset_len = len(self.videos)
        # random.shuffle(self.videos)

    def preload_deca_bbox(self, ):
        bbox_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_deca_bbox.pt')

        if os.path.exists(bbox_path):
            self.deca_bboxes = torch.load(bbox_path)
        else:
            print('pre-loading deca bboxes...')
            self.deca_bboxes = {}
            for item in tqdm(self.videos):
                bbox = torch.load(os.path.join(item, 'deca_bbox.pt')) * self.lmk_scale
                self.deca_bboxes[os.path.basename(item)] = bbox
            torch.save(self.deca_bboxes, bbox_path)

    def preload_arcface_bbox(self, ):
        bbox_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_arcface_bbox.pt')

        if os.path.exists(bbox_path):
            self.arcface_bboxes = torch.load(bbox_path)
        else:
            print('pre-loading arcface bboxes...')
            self.arcface_bboxes = {}
            for item in tqdm(self.videos):
                bbox = torch.load(os.path.join(item, 'arcface_bbox.pt')) * self.lmk_scale
                self.arcface_bboxes[os.path.basename(item)] = bbox
            torch.save(self.arcface_bboxes, bbox_path)

    def preload_arcface_M(self, ):
        M_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_arcface_M.pt')
        self.arcface_M = torch.load(M_path)

    def __getitem__(self, idx):
        '''
        return:{
            'img_source': img_source, 
            'lmk_source': lmk_source, 
            'img_gt': img_gt, 
            'lmk_gt': lmk_gt, 
            'img_drive': img_drive, 
            'lmk_drive': lmk_drive
        }
        '''
        video_path = self.videos[idx]
        key_name = os.path.basename(video_path)

        # fast loading
        lengths = len(self.deca_bboxes[key_name])
        ij = np.random.randint(0, lengths, size=2)
        while abs(ij[0] - ij[1]) <= 30:
            ij = np.random.randint(0, lengths, size=2)

        two_paths = [os.path.join(video_path, '%.7d.png' % _i) for _i in ij]

        img_source = Image.open(two_paths[0]).convert('RGB')
        M_source_deca = warp_func.bbox2AffineMatrix(self.deca_bboxes[key_name][ij[0]], size=self.img_size_for_deca)
        img_gt = Image.open(two_paths[1]).convert('RGB')
        M_gt_deca = warp_func.bbox2AffineMatrix(self.deca_bboxes[key_name][ij[1]], size=self.img_size_for_deca)


        if self.use_bbox:
            M_source_arcface = warp_func.bbox2AffineMatrix(self.arcface_bboxes[key_name][ij[0]], size=self.img_size_for_arcface)
            M_gt_arcface = warp_func.bbox2AffineMatrix(self.arcface_bboxes[key_name][ij[1]], size=self.img_size_for_arcface)
        else:
            M_source_arcface = self.arcface_M[key_name][ij[0]]
            M_gt_arcface = self.arcface_M[key_name][ij[1]]

        

        # frames_paths = sorted(glob.glob(video_path + '/*.png'))
        # nframes = len(frames_paths)
        # items = random.sample(self.index_list[:nframes], 2)

        # img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        # M_source_deca = warp_func.bbox2AffineMatrix(self.deca_bboxes[key_name][items[0]], size=self.img_size_for_deca)
        # M_source_arcface = warp_func.bbox2AffineMatrix(self.arcface_bboxes[key_name][items[0]], size=self.img_size_for_arcface)

        # img_gt = Image.open(frames_paths[items[1]]).convert('RGB')
        # M_gt_deca = warp_func.bbox2AffineMatrix(self.deca_bboxes[key_name][items[1]], size=self.img_size_for_deca)
        # M_gt_arcface = warp_func.bbox2AffineMatrix(self.arcface_bboxes[key_name][items[1]], size=self.img_size_for_arcface)

        # if self.augmentation:
        #     img_source, img_gt = self.aug(img_source, img_gt)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_gt = self.transform(img_gt)

        driving_id = np.random.randint(0, self.dataset_len)
        info_drive = self.load_drive_frame(driving_id)
        
        sample = {
            'img_source': img_source, 'M_source_deca': M_source_deca, 'M_source_arcface': M_source_arcface, 
            'img_gt': img_gt, 'M_gt_deca': M_gt_deca, 'M_gt_arcface': M_gt_arcface
            }
        sample.update(info_drive)

        return sample

    def load_drive_frame(self, idx):
        # load driving image of another identity
        video_path = self.videos[idx]
        key_name = os.path.basename(video_path)

        # fast loading
        lengths = len(self.deca_bboxes[key_name])
        ij = random.randint(0, lengths-1)
        img_path = os.path.join(video_path, '%.7d.png' % ij)

        img_drive = Image.open(img_path).convert('RGB')        
        M_drive_deca = warp_func.bbox2AffineMatrix(self.deca_bboxes[key_name][ij], size=self.img_size_for_deca)
        if self.use_bbox:
            M_drive_arcface = warp_func.bbox2AffineMatrix(self.arcface_bboxes[key_name][ij], size=self.img_size_for_arcface)
        else:
            M_drive_arcface = self.arcface_M[key_name][ij]    
        
        # drive_frame_paths = sorted(glob.glob(video_path + '/*.png'))
        # item = random.sample(self.index_list[:len(drive_frame_paths)], 1)
        
        # img_drive = Image.open(drive_frame_paths[item[0]]).convert('RGB')        
        # M_drive_deca = warp_func.bbox2AffineMatrix(self.deca_bboxes[key_name][item[0]], size=self.img_size_for_deca)
        # M_drive_arcface = warp_func.bbox2AffineMatrix(self.arcface_bboxes[key_name][item[0]], size=self.img_size_for_arcface)

        if self.transform is not None:
            img_drive = self.transform(img_drive)
        return {'img_drive': img_drive, 'M_drive_deca': M_drive_deca, 'M_drive_arcface': M_drive_arcface}

    def __len__(self):
        return len(self.videos)



class Vox256_slow(Dataset):
    def __init__(self, split, size=256, transform=None, augmentation=False):
        
        self.split = split
        if split == 'train':
            self.ds_path = '/home/ps/workspace/HDD/lingjun/dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/train'
            self.is_train = True
        elif split == 'test':
            self.ds_path = '/home/ps/workspace/HDD/lingjun/dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/test'
            self.is_train = False
        else:
            raise NotImplementedError
            
        assert os.path.exists(self.ds_path)

        videos_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_videos.pt')
        if os.path.exists(videos_path):
            print('load pre-defined video names')
            self.videos = torch.load(videos_path)
        else:
            self.videos = sorted(glob.glob(os.path.join(self.ds_path, 'id*'))) # os.listdir(self.ds_path)
            torch.save(self.videos, videos_path)
        self.augmentation = augmentation
        self.aug = AugmentationTransform(False, False, True)

        self.transform = transform
        
        self.preload_lm68p()
        self.lmk_scale = size / 320
        self.index_list = list(range(5000))
        # self.preload_3d_shape_params()

        self.vshift_scale_for_arcface = 0.12
        self.img_size_for_arcface = 112

        self.vshift_scale_for_deca = 0.06
        self.img_size_for_deca = 224

    def preload_lm68p(self, ):
        landmarks_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_landmarks.pt')

        if os.path.exists(landmarks_path):
            self.landmarks = torch.load(landmarks_path)
        else:
            print('pre-loading landmarks...')
            self.landmarks = {}
            for item in tqdm(self.videos):
                ldmks = torch.load(os.path.join(item, 'landmarks2d.pt')) * self.lmk_scale
                self.landmarks[os.path.basename(item)] = ldmks
            torch.save(self.landmarks, landmarks_path)
        # self.gen_width_height_ratio_matrix()
    
    def gen_width_height_ratio_matrix(self, ):
        # ratio_file_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_wh_ratio_index.pt')

        # if os.path.exists(ratio_file_path):
        #     self.wh_ratio_dist_index = torch.load(ratio_file_path)
        # else:
        #     print('pre-loading face width-height ratio...')
        keys = list(self.landmarks.keys())
        wh_ratio_array = []
        for item in keys:
            lm68p = self.landmarks[item]
            a = lm68p[:, 1] - lm68p[:, 15]  # (bs, 2)
            b = lm68p[:, 27] - lm68p[:, 57] # (bs, 2)
            dist = torch.mean(torch.sqrt(torch.sum(a**2, dim=1, ) / torch.sum(b**2, dim=1, )))
            wh_ratio_array.append(dist)
        wh_ratio_array = torch.Tensor(wh_ratio_array)    # (N,)
        dist = wh_ratio_array - wh_ratio_array.unsqueeze(1) # (N, N)
        self.wh_ratio_dist_index = {}
        for i, k in enumerate(tqdm(keys)):
            self.wh_ratio_dist_index[k] = torch.argsort(dist[i], descending=True)
        # torch.save(self.wh_ratio_dist_index, ratio_file_path)
        self.topK = len(self.wh_ratio_dist_index) // 5 if self.is_train else len(self.wh_ratio_dist_index) // 10

    def preload_3d_shape_params(self, ):
        shape_params_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_3dparams.pt')

        if os.path.exists(shape_params_path):
            self.shape_params = torch.load(shape_params_path)
        else:
            print('pre-loading 3d shape parameters...')
            self.shape_params = {}
            for item in tqdm(self.videos):
                params = torch.load(os.path.join(item, '3dparams.pt'))
                self.shape_params[os.path.basename(item)] = params
            torch.save(self.shape_params, shape_params_path)
        self.gen_shape_distance_matrix()

    def gen_shape_distance_matrix(self, ):
        keys = list(self.shape_params.keys())
        shape_array = []
        for k in keys:
            shape_array.append(self.shape_params[k]['shape'][0, :20])
        shape_array = torch.stack(shape_array)
        self.shape_dist_index = {}
        # print(dist.shape)
        for i, k in enumerate(tqdm(keys)):
            dist = torch.square(shape_array[i] - shape_array).sum(1)
            self.shape_dist_index[k] = torch.argsort(dist, descending=True) # index, (N, )
        self.topK = len(self.shape_dist_index) // 5 if self.is_train else len(self.shape_dist_index) // 10
        
    def __getitem__(self, idx):
        '''
        return:{
            'img_source': img_source, 
            'lmk_source': lmk_source, 
            'img_gt': img_gt, 
            'lmk_gt': lmk_gt, 
            'img_drive': img_drive, 
            'lmk_drive': lmk_drive
        }
        '''
        video_path = self.videos[idx]
        key_name = os.path.basename(video_path)
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)
        items = random.sample(self.index_list[:nframes], 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        lmk_source = self.landmarks[key_name][items[0]] #* self.lmk_scale
        M_source_deca = warp_func.estimate_single_transform_torch(lmk_source, size=self.img_size_for_deca, vshift_scale=self.vshift_scale_for_deca)
        M_source_arcface = warp_func.estimate_single_transform_torch(lmk_source, size=self.img_size_for_arcface, vshift_scale=self.vshift_scale_for_arcface)

        img_gt = Image.open(frames_paths[items[1]]).convert('RGB')
        lmk_gt = self.landmarks[key_name][items[1]] #* self.lmk_scale
        M_gt_deca = warp_func.estimate_single_transform_torch(lmk_gt, size=self.img_size_for_deca, vshift_scale=self.vshift_scale_for_deca)
        M_gt_arcface = warp_func.estimate_single_transform_torch(lmk_gt, size=self.img_size_for_arcface, vshift_scale=self.vshift_scale_for_arcface)

        if self.augmentation:
            img_source, img_gt = self.aug(img_source, img_gt)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_gt = self.transform(img_gt)
        
        info_drive = self.load_drive_frame(random.randint(0, len(self.videos)-1))
        # img_drive, lmk_drive = self.load_drive_frame(random.choice(self.wh_ratio_dist_index[key_name][:self.topK]))
        
        sample = {'img_source': img_source, 'lmk_source': lmk_source, 'M_source_deca': M_source_deca, 'M_source_arcface': M_source_arcface, 'img_gt': img_gt, 'lmk_gt': lmk_gt, 'M_gt_deca': M_gt_deca, 'M_gt_arcface': M_gt_arcface}
        # sample['img_drive'] = img_drive
        # sample['lmk_drive'] = lmk_drive
        sample.update(info_drive)
        return sample

    def load_drive_frame(self, idx):
        # load driving image of another identity
        video_path = self.videos[idx]
        drive_frame_paths = sorted(glob.glob(video_path + '/*.png'))
        item = random.sample(self.index_list[:len(drive_frame_paths)], 1)
        
        img_drive = Image.open(drive_frame_paths[item[0]]).convert('RGB')
        lmk_drive = self.landmarks[os.path.basename(video_path)][item[0]] #* self.lmk_scale
        
        M_drive_deca = warp_func.estimate_single_transform_torch(lmk_drive, size=self.img_size_for_deca, vshift_scale=self.vshift_scale_for_deca)
        M_drive_arcface = warp_func.estimate_single_transform_torch(lmk_drive, size=self.img_size_for_arcface, vshift_scale=self.vshift_scale_for_arcface)

        if self.transform is not None:
            img_drive = self.transform(img_drive)
        return {'img_drive': img_drive, 'lmk_drive': lmk_drive, 'M_drive_deca': M_drive_deca, 'M_drive_arcface': M_drive_arcface}

    def __len__(self):
        return len(self.videos)


class Vox256Plus(Dataset):
    def __init__(self, split, size=256, transform=None, augmentation=False):
        
        self.split = split
        if split == 'train':
            self.ds_path = '/home/ps/workspace/HDD/lingjun/dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/train'
            self.is_train = True
        elif split == 'test':
            self.ds_path = '/home/ps/workspace/HDD/lingjun/dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/test'
            self.is_train = False
        else:
            raise NotImplementedError
            
        assert os.path.exists(self.ds_path)

        videos_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_videos.pt')
        if os.path.exists(videos_path):
            print('load pre-defined video names')
            self.videos = torch.load(videos_path)
        else:
            self.videos = sorted(glob.glob(os.path.join(self.ds_path, 'id*'))) # os.listdir(self.ds_path)
            torch.save(self.videos, videos_path)
        self.augmentation = augmentation
        self.aug = AugmentationTransform(False, False, True)

        self.lmk_scale = size / 320
        self.transform = transform
        self.preload_lm68p()
        self.index_list = list(range(5000))
        # self.preload_3d_shape_params()

        self.vshift_scale_for_arcface = 0.12
        self.img_size_for_arcface = 112

        self.vshift_scale_for_deca = 0.06
        self.img_size_for_deca = 224

    def preload_lm68p(self, ):
        landmarks_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_landmarks.pt')

        if os.path.exists(landmarks_path):
            self.landmarks = torch.load(landmarks_path)
        else:
            print('pre-loading landmarks...')
            self.landmarks = {}
            for item in tqdm(self.videos):
                ldmks = torch.load(os.path.join(item, 'landmarks2d.pt')) * self.lmk_scale
                self.landmarks[os.path.basename(item)] = ldmks
            torch.save(self.landmarks, landmarks_path)
        # self.gen_width_height_ratio_matrix()
    
    def gen_width_height_ratio_matrix(self, ):
        # ratio_file_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_wh_ratio_index.pt')

        # if os.path.exists(ratio_file_path):
        #     self.wh_ratio_dist_index = torch.load(ratio_file_path)
        # else:
        #     print('pre-loading face width-height ratio...')
        keys = list(self.landmarks.keys())
        wh_ratio_array = []
        for item in keys:
            lm68p = self.landmarks[item]
            a = lm68p[:, 1] - lm68p[:, 15]  # (bs, 2)
            b = lm68p[:, 27] - lm68p[:, 57] # (bs, 2)
            dist = torch.mean(torch.sqrt(torch.sum(a**2, dim=1, ) / torch.sum(b**2, dim=1, )))
            wh_ratio_array.append(dist)
        wh_ratio_array = torch.Tensor(wh_ratio_array)    # (N,)
        dist = wh_ratio_array - wh_ratio_array.unsqueeze(1) # (N, N)
        self.wh_ratio_dist_index = {}
        for i, k in enumerate(tqdm(keys)):
            self.wh_ratio_dist_index[k] = torch.argsort(dist[i], descending=True)
        # torch.save(self.wh_ratio_dist_index, ratio_file_path)
        self.topK = len(self.wh_ratio_dist_index) // 5 if self.is_train else len(self.wh_ratio_dist_index) // 10

    def preload_3d_shape_params(self, ):
        shape_params_path = os.path.join(os.path.dirname(self.ds_path), f'{self.split}_3dparams.pt')

        if os.path.exists(shape_params_path):
            self.shape_params = torch.load(shape_params_path)
        else:
            print('pre-loading 3d shape parameters...')
            self.shape_params = {}
            for item in tqdm(self.videos):
                params = torch.load(os.path.join(item, '3dparams.pt'))
                self.shape_params[os.path.basename(item)] = params
            torch.save(self.shape_params, shape_params_path)
        self.gen_shape_distance_matrix()

    def gen_shape_distance_matrix(self, ):
        keys = list(self.shape_params.keys())
        shape_array = []
        for k in keys:
            shape_array.append(self.shape_params[k]['shape'][0, :20])
        shape_array = torch.stack(shape_array)
        self.shape_dist_index = {}
        # print(dist.shape)
        for i, k in enumerate(tqdm(keys)):
            dist = torch.square(shape_array[i] - shape_array).sum(1)
            self.shape_dist_index[k] = torch.argsort(dist, descending=True) # index, (N, )
        self.topK = len(self.shape_dist_index) // 5 if self.is_train else len(self.shape_dist_index) // 10
        
    def gen_warpping_matrix(self, ):
        self.M_deca = {}
        self.M_arcface = {}
        for k, lmks in tqdm(self.landmarks.items()):
            self.M_deca[k] = warp_func.estimate_transform_torch(lmks, size=self.img_size_for_deca, vshift_scale=self.vshift_scale_for_deca)
            self.M_arcface[k] = warp_func.estimate_transform_torch(lmks, size=self.img_size_for_arcface, vshift_scale=self.vshift_scale_for_arcface)

    def __getitem__(self, idx):
        '''
        return:{
            'img_source': img_source, 
            'lmk_source': lmk_source, 
            'img_gt': img_gt, 
            'lmk_gt': lmk_gt, 
            'img_drive': img_drive, 
            'lmk_drive': lmk_drive
        }
        '''
        video_path = self.videos[idx]
        key_name = os.path.basename(video_path)
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)
        items = random.sample(self.index_list[:nframes], 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        # lmk_source = self.landmarks[key_name][items[0]] #* self.lmk_scale
        M_source_deca = self.M_deca[key_name][items[0]]
        M_source_arcface = self.M_arcface[key_name][items[0]]

        img_gt = Image.open(frames_paths[items[1]]).convert('RGB')
        # lmk_gt = self.landmarks[key_name][items[1]] #* self.lmk_scale
        M_gt_deca = self.M_deca[key_name][items[1]]
        M_gt_arcface = self.M_arcface[key_name][items[0]]

        if self.augmentation:
            img_source, img_gt = self.aug(img_source, img_gt)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_gt = self.transform(img_gt)
        
        info_drive = self.load_drive_frame(random.randint(0, len(self.videos)-1))
        # img_drive, lmk_drive = self.load_drive_frame(random.choice(self.wh_ratio_dist_index[key_name][:self.topK]))
        
        sample = {
            'img_source': img_source, 
            # 'lmk_source': lmk_source, 
            'M_source_deca': M_source_deca, 
            'M_source_arcface': M_source_arcface, 
            'img_gt': img_gt, 
            # 'lmk_gt': lmk_gt, 
            'M_gt_deca': M_gt_deca, 
            'M_gt_arcface': M_gt_arcface
        }
        # sample['img_drive'] = img_drive
        # sample['lmk_drive'] = lmk_drive
        sample.update(info_drive)
        return sample

    def load_drive_frame(self, idx):
        # load driving image of another identity
        video_path = self.videos[idx]
        key_name = os.path.basename(video_path)
        drive_frame_paths = sorted(glob.glob(video_path + '/*.png'))
        item = random.sample(self.index_list[:len(drive_frame_paths)], 1)
        
        img_drive = Image.open(drive_frame_paths[item[0]]).convert('RGB')
        # lmk_drive = self.landmarks[key_name][item[0]] #* self.lmk_scale
        
        M_drive_deca = self.M_deca[key_name][item[0]]
        M_drive_arcface = self.M_arcface[key_name][item[0]]

        if self.transform is not None:
            img_drive = self.transform(img_drive)
        return {
            'img_drive': img_drive, 
            # 'lmk_drive': lmk_drive, 
            'M_drive_deca': M_drive_deca, 
            'M_drive_arcface': M_drive_arcface
        }

    def __len__(self):
        return len(self.videos)


class Vox256_vox2german(Dataset):
    def __init__(self, transform=None):
        self.source_root = './datasets/german/'
        self.driving_root = './datasets/vox/test/'

        self.anno = pd.read_csv('pairs_annotations/german_vox.csv')

        self.source_imgs = os.listdir(self.source_root)
        self.transform = transform

    def __getitem__(self, idx):
        source_name = str('%03d' % self.anno['source'][idx])
        driving_name = self.anno['driving'][idx]

        source_vid_path = self.source_root + source_name
        driving_vid_path = self.driving_root + driving_name

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.source_imgs)


class Vox256_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Vox256_cross(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.anno = pd.read_csv('pairs_annotations/vox256.csv')
        self.transform = transform

    def __getitem__(self, idx):
        source_name = self.anno['source'][idx]
        driving_name = self.anno['driving'][idx]

        source_vid_path = os.path.join(self.ds_path, source_name)
        driving_vid_path = os.path.join(self.ds_path, driving_name)

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.videos)


class Taichi(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/taichi/train/'
        else:
            self.ds_path = './datasets/taichi/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(True, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):

        video_path = self.ds_path + self.videos[idx]
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Taichi_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/taichi/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class TED(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/ted/train/'
        else:
            self.ds_path = './datasets/ted/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class TED_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/ted/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Vox256TEST(Dataset):
    def __init__(self, split, size=256, transform=None, augmentation=False):
        
        self.split = split
        if split == 'train':
            self.ds_path = '../../dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/train'
            self.is_train = True
        elif split == 'test':
            self.ds_path = '../../dataset/VoxCeleb1-HQ/imgs-voxceleb-25fps/test'
            self.is_train = False
        else:
            raise NotImplementedError
            
        self.transform = transform
        # assert os.path.exists(self.ds_path)

        self.vshift_scale_for_arcface = 0.12
        self.img_size_for_arcface = 112

        self.vshift_scale_for_deca = 0.06
        self.img_size_for_deca = 224

        self.dataset_len = 18320
        # random.shuffle(self.videos)

    def __getitem__(self, idx):
        '''
        return:{
            'img_source': img_source, 
            'lmk_source': lmk_source, 
            'img_gt': img_gt, 
            'lmk_gt': lmk_gt, 
            'img_drive': img_drive, 
            'lmk_drive': lmk_drive
        }
        '''
        img_source = Image.fromarray((np.random.randn(320, 320, 3)*255).astype(np.uint8))
        M_source_deca = warp_func.bbox2AffineMatrix([5, 20, 180, 180], size=self.img_size_for_deca)
        M_source_arcface = warp_func.bbox2AffineMatrix([5, 20, 180, 180], size=self.img_size_for_arcface)

        img_gt = Image.fromarray((np.random.randn(320, 320, 3)*255).astype(np.uint8))
        M_gt_deca = warp_func.bbox2AffineMatrix([5, 20, 180, 180], size=self.img_size_for_deca)
        M_gt_arcface = warp_func.bbox2AffineMatrix([5, 20, 180, 180], size=self.img_size_for_arcface)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_gt = self.transform(img_gt)

        driving_id = np.random.randint(0, self.dataset_len)
        # driving_name = os.path.basename(self.videos[driving_id])
        # while driving_name[:20] == key_name[:20] and self.is_train:
        #     print(driving_name, key_name, 'sampling another driving frame')
        #     driving_id = random.randint(0, self.dataset_len - 1)
        info_drive = self.load_drive_frame(driving_id)
        
        sample = {
            'img_source': img_source, 'M_source_deca': M_source_deca, 'M_source_arcface': M_source_arcface, 
            'img_gt': img_gt, 'M_gt_deca': M_gt_deca, 'M_gt_arcface': M_gt_arcface
            }
        sample.update(info_drive)

        return sample

    def load_drive_frame(self, idx):
        img_drive = Image.fromarray((np.random.randn(320, 320, 3)*255).astype(np.uint8))
        M_drive_deca = warp_func.bbox2AffineMatrix([5, 20, 180, 180], size=self.img_size_for_deca)
        M_drive_arcface = warp_func.bbox2AffineMatrix([5, 20, 180, 180], size=self.img_size_for_arcface)

        if self.transform is not None:
            img_drive = self.transform(img_drive)
        return {'img_drive': img_drive, 'M_drive_deca': M_drive_deca, 'M_drive_arcface': M_drive_arcface}

    def __len__(self):
        return 18320

if __name__ == "__main__":
    import torchvision
    size = 256
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )
    dataset = Vox256('train', size, transform, False)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=4,
        pin_memory=True,
        drop_last=True,
    )
    for batch in tqdm(loader):
        continue
