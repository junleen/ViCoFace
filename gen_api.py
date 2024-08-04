import torch
import torch.nn as nn
from networks.generator import Generator_Cycle
import numpy as np
import torchvision
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import glob
from moviepy.editor import ImageSequenceClip
from PIL import Image
from skimage import io
from skimage import img_as_ubyte
import imageio
import cv2
from face_det.tracker.face_tracker import FaceTracker
from utils.warp_func import cv2CropWarpAffine


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def save_video(save_path, name, vid_target_recon, fps=10, save_frames=False):
    vid = (vid_target_recon.permute(0, 2, 3, 4, 1) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).cpu() # [1, 25, 256, 256, 3]
    torchvision.io.write_video(save_path + '%s.mp4' % name, vid[0], fps=fps)
    if save_frames:
        os.makedirs(os.path.join(save_path + name), exist_ok=True)
        for i in range(vid.shape[1]):
            io.imsave(os.path.join(save_path + name, '%03d.png' % (i)), vid[0, i].numpy()) # .permute(1, 2, 0)


def data_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm

def extract_frames_from_a_video(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    frames = np.stack(frames)   # (N, H, W, 3)
    return frames, fps

class Gen_API(nn.Module):
    def __init__(self, dataset='vox', size=256, device='cuda:0', save_path='./id_explore', ckpt_path=None, motion_dim=20, id_dim=10):
        super(Gen_API, self).__init__()

        self.transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        self.device = device
        self.motion_dim = motion_dim
        self.id_dim = id_dim

        if dataset == 'vox':
            if ckpt_path is not None:
                path = ckpt_path
            else:
                path = 'checkpoints/660000.pt'
            print('load from:', path)
        else:
            raise NotImplementedError
            # pass

        self.save_path = os.path.join(save_path)
        os.makedirs(self.save_path, exist_ok=True)

        print('==> loading model')
        self.gen = Generator_Cycle(size=size, style_dim=512, motion_dim=motion_dim, id_dim=id_dim, channel_multiplier=1, blur_kernel=[1,3,3,1]).to(self.device)
        weight = torch.load(path, map_location='cpu')['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        self.face_tracker = FaceTracker(device=device)

    
    @torch.no_grad()
    def get_encoding(self, src):
        input_src = self.transform(src).unsqueeze(0).to(self.device)
        code_dict, _ = self.gen.enc.forward(input_src, input_drive=None)
        wa = code_dict['wa'].cpu().numpy() # (N, 512)
        a = code_dict['h_motion'].cpu().numpy() # (N, 20)
        b = code_dict['h_id'].cpu().numpy() # (N, 10)
        return wa, a, b

    def crop_roi(self, image, bbox, expand_scale=1.6):
        x1, y1, x2, y2 = bbox.astype(np.int32)
        center = (x1+x2)/2, (y1+y2)/2
        width = height = (x2-x1 + y2-y1)/2*expand_scale
        crop_box = np.array([center[0]-width/2, center[1]-height/2, width, height]).astype(np.int32)
        roi = cv2CropWarpAffine(image, crop_box, crop_size=256, interpolate=cv2.INTER_CUBIC)
        return roi

    @torch.no_grad()
    def reenact(self, src, drv, same_id, crop_image=False):
        '''
        Args:
            src: np.ndarray (H, W, 3)
            drv: np.ndarray (H, W, 3)
            same_id: bool, if the source and driving images are the same person
            crop_image: bool, if crop the face region. Note that the too big or too small face region ratio will introduce unexpected artifacts人脸需要占人脸的比例为0.5-0.8，否则会出现异常
        Return:
            img_recon: np.ndarray (H, W, 3)        
        '''
        if crop_image:
            detected_faces = self.face_tracker.face_detector(src, rgb=True)
            src = self.crop_roi(src, detected_faces[0][:4])
            detected_faces = self.face_tracker.face_detector(drv, rgb=True)
            drv = self.crop_roi(drv, detected_faces[0][:4])

        input_src = self.transform(src).unsqueeze(0).to(self.device)
        input_drv = self.transform(drv).unsqueeze(0).to(self.device)
        img_recon = self.gen.inference(input_src, input_drv, same_id=same_id)
        img_recon = (img_recon['img_reenacted'].clamp(-1, 1)[0].permute(1,2,0).cpu().numpy()*127.5 + 127.5).astype(np.uint8)
        return img_recon
    
    def save_img(self, img: np.ndarray, img_name: str):
        os.makedirs(self.save_path, exist_ok=True)
        io.imsave(os.path.join(self.save_path, img_name), img)

    def video_reenactment(self, source_image, driving_video_path, crop_image=False):

        print('==> running')
        with torch.no_grad():
            predictions = []
            source = self.transform(source_image).unsqueeze(0).to(self.device).float()
            code_source, _ = self.gen.enc(source, input_drive=None)
            driving, fps = extract_frames_from_a_video(driving_video_path)
            if crop_image:
                
                detected_bboxes = self.face_tracker.face_detector(source_image, rgb=True)
                source_image = self.crop_roi(source_image, detected_bboxes[0][:4])
                io.imsave(os.path.join(self.save_path, 'source.jpg'), source_image)
                source = self.transform(source_image).unsqueeze(0).to(self.device).float()
                code_source, _ = self.gen.enc(source, input_drive=None)

                detected_bboxes = self.face_tracker.face_detector(driving[0], rgb=True)
                drive_images = []
                for i in range(driving.shape[0]):
                    drive_crop = self.crop_roi(driving[i], detected_bboxes[0][:4])
                    drive_images.append(drive_crop)
                driving = np.stack(drive_images)

            for frame_idx in tqdm(range(driving.shape[0])):
                driving_frame = self.transform(driving[frame_idx]).unsqueeze(0).to(self.device)
                code_drive, _ = self.gen.enc.forward(driving_frame, input_drive=None)
                alpha = [code_drive['h_motion'], code_source['h_id']]
                img = self.gen.dec(code_source['wa'], alpha, code_source['feats'])
                predictions.append(img.clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()*0.5 + 0.5) # (-1, 1)

        name = os.path.basename(src_path) + 'animated-res-' + os.path.basename(driving_video_path).rsplit('.', 1)[0]
        imageio.mimsave(
            os.path.join(self.save_path, '%s.mp4' % name), 
            [np.hstack([cv2.resize(source_image, (256, 256)), driving[i], img_as_ubyte(frame)]) for i, frame in enumerate(predictions)], fps=fps)
        print('saved results in', name)
        return predictions

if __name__ == '__main__':
    size = 256
    device = 'cuda:0'
    ckpt_path = 'checkpoints/660000.pt'
    save_path = './res_manipulation'
    
    demo = Gen_API(dataset='vox', save_path=save_path, size=size, device=device, ckpt_path=ckpt_path,  motion_dim=20, id_dim=10)
    
    src_path = 'data/happy_level2_0.jpg'
    drv_path = 'data/driving2.mp4'
    img_src = np.array(Image.open(src_path).convert('RGB'))

    if drv_path.rsplit('.', 1)[1] in {'jpg', 'png', 'bmp', 'jpeg'}:
        save_img_path = os.path.join(demo.save_path, '%s-%s' % (os.path.basename(src_path), os.path.basename(drv_path)))
        img_drv = np.array(Image.open(drv_path).convert('RGB'))
        img_recon = demo.reenact(img_src, img_drv, same_id=True, crop_image=True)
        saved_result = np.hstack([cv2.resize(img_src, (size, size)), cv2.resize(img_drv, (size, size)), cv2.resize(img_recon, (size, size))])
        io.imsave(save_img_path, saved_result)
    elif drv_path.rsplit('.', 1)[1] in {'mp4', 'avi', 'mov'}:
        demo.video_reenactment(img_src, drv_path, crop_image=True)
