import torch
from torch import nn
import torch.nn.functional as F
from .encoder import Encoder
from .styledecoder import Synthesis


class HighPass(nn.Module):
    def __init__(self, w_hpf=1):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]).unsqueeze(0).unsqueeze(1) / w_hpf)

    def forward(self, x):
        filter = self.filter.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

class EdgeLoss(nn.Module):
    def __init__(self, device='cuda', statistical=False):
        super(EdgeLoss, self).__init__()
        self.conv = HighPass(1).to(device)
        self.conv.eval()
        self.statistical = statistical
    
    def forward(self, input, reference):
        out = self.conv(input)
        ref = self.conv(reference)
        if self.statistical:
            loss = torch.abs(torch.abs(out).mean() - torch.abs(ref).mean())
        else:
            loss = F.l1_loss(out, ref)
        
        return loss

class Generator_Cycle(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, id_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_Cycle, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim, id_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, id_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def self_reconstruct(self, img_source):
        wa, alpha, feats = self.enc.self_enc(img_source)
        img = self.synthesis(wa, alpha, feats)
        return img

    def forward(self, img_source, img_gt, img_drive=None, inference=False, cyclic=False):
        '''
        args:
            img_source: source image of id1
            img_gt: another image of id1
            img_drive: driving image of id2
        return:
            code_source: code_source, 
            code_gt: code_gt, 
            img_src_self_gen: img_src_self_gen
        (optional):
            code_drive: code_drive
            img_src_cross_gen: img_src_cross_gen
            img_src_drive_gen: img_src_drive_gen
        '''
        if inference:
            _tmp1 = self.inference(img_source=img_source, img_drive=img_gt, self_reconstruction=False, same_id=True)
            output_dict = {
                'img_source': img_source, 
                'img_gt': img_gt, 
                'img_src_self_gen': _tmp1['img_reenacted'], 
                # 'img_src_self_reconstruction': _tmp1['img_src_self_reconstruction'], 
                # 'img_normal': _tmp1['img_normal'],
            }
            
            if img_drive is not None:
                _tmp2 = self.inference(img_source=img_source, img_drive=img_drive, self_reconstruction=False, same_id=False)
                output_dict['img_drive'] = img_drive 
                output_dict['img_src_cross_gen'] = _tmp2['img_reenacted']
            return output_dict
        
        code_source, code_gt = self.enc(img_source, img_gt)
        # using the motion from gt image and the id from source image, to synthesize self_gen image
        alpha_gts = [code_gt['h_motion'], code_gt['h_id']]
        img_src_self_gen = self.dec(code_source['wa'], alpha_gts, code_source['feats']) # as close img_gt as possible
        output_dict = {
            'code_source': code_source, 
            'code_gt': code_gt, 
            'img_src_self_gen': img_src_self_gen,
        }

        if img_drive is not None:
            code_drive, _ = self.enc(img_drive, None)
            # using the motion from drive image and the id from source image, to synthesize cross_gen image
            alpha_ds = [code_drive['h_motion'], code_source['h_id']]
            img_src_cross_gen = self.dec(code_source['wa'], alpha_ds, code_source['feats']) # motion like drive image, id like source image
            output_dict['code_drive'] = code_drive
            output_dict['img_src_cross_gen'] = img_src_cross_gen
            
            # cycle reconstruction
            if cyclic:
                code_src_cross_gen, code_src_self_gen = self.enc(img_src_cross_gen, img_src_self_gen)
                alpha_cycle = [code_src_self_gen['h_motion'], code_src_cross_gen['h_id']]
                img_cycle_gen = self.dec(code_src_cross_gen['wa'], alpha_cycle, code_src_cross_gen['feats'])
                output_dict['img_src_cycle_gen'] = img_cycle_gen
            
        return output_dict

    def inference(self, img_source, img_drive=None, h_start=None, self_reconstruction=False, same_id=False):
        code_source, code_drive = self.enc(img_source, img_drive)
        '''
        cases of alpha:
        case 1: None, to normalized face, 
        case 2: [h_motion, h_id], concatenate
        case 3: [h_motion_target, h_motion_start, h_motion_source, h_id_source], concatenate (target - start + source, id_source)
        '''
        if img_drive is not None:
            if h_start is not None:
                alpha = [code_drive['h_motion'], h_start, code_source['h_motion'], code_source['h_id']]
            else:
                if same_id:
                    alpha = [code_drive['h_motion'], code_drive['h_id']]
                else:
                    alpha = [code_drive['h_motion'], code_source['h_id']]
        else:
            alpha = None
        output_dict = {}
        img_reenacted = self.dec(code_source['wa'], alpha, code_source['feats'])
        output_dict['img_reenacted'] = img_reenacted
        
        if self_reconstruction:
            img_normal = self.dec(code_source['wa'], [torch.zeros_like(code_source['h_motion']), code_source['h_id']], code_source['feats'])
            img_motion = self.dec(code_source['wa'], [code_source['h_motion'], torch.zeros_like(code_source['h_id'])], code_source['feats'])
            img_src_self_reconstruction = self.dec(code_source['wa'], [code_source['h_motion'], code_source['h_id']], code_source['feats'])
            output_dict['img_normal'] = img_normal
            output_dict['img_motion'] = img_motion
            output_dict['img_src_self_reconstruction'] = img_src_self_reconstruction

        return output_dict

