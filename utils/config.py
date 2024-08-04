'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse

cfg = CN()

# ---------------------------------------------------------------------------- #
# Options for args
# ---------------------------------------------------------------------------- #
cfg.args = CN()
cfg.args.iter = 800000
cfg.args.size = 256
cfg.args.batch_size = 32
cfg.args.d_reg_every = 16
cfg.args.g_reg_every = 16
cfg.args.lr = 2e-3
cfg.args.channel_multiplier = 1
cfg.args.start_iter = 0
cfg.args.latent_dim_style = 512
cfg.args.latent_dim_motion = 20
cfg.args.latent_dim_id = 10
cfg.args.dataset = 'vox'
cfg.args.exp_path = './exps/'
cfg.args.exp_name = 'v1'


# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.L1 = 15
cfg.loss.id = 6
cfg.loss.lmk = 3
cfg.loss.useWlmk = True
cfg.loss.use_cycle = True
cfg.loss.eyed = 0.
cfg.loss.lipd = 0.
cfg.loss.pose = 50
cfg.loss.photo = 2.0
cfg.loss.useSeg = True
cfg.loss.gan_g = 0.5
cfg.loss.gan_cross_g = 0.5
cfg.loss.cross_gt_L1 = 20
cfg.loss.gan_cycle_g = 0
cfg.loss.cycle_vgg = 1
cfg.loss.edge = 5
cfg.loss.gan_d = 1
cfg.loss.id_shape_only = True
cfg.loss.reg_shape = 1e-04
cfg.loss.reg_exp = 1e-04
cfg.loss.reg_tex = 1e-04
cfg.loss.reg_light = 1.
cfg.loss.reg_jaw_pose = 0. #1.
cfg.loss.use_gender_prior = False
cfg.loss.shape_consistency = True
# loss for detail
cfg.loss.detail_consistency = True
cfg.loss.useConstraint = True
cfg.loss.mrf = 5e-2
cfg.loss.photo_D = 2.
cfg.loss.reg_sym = 0.005
cfg.loss.reg_z = 0.005
cfg.loss.reg_diff = 0.005


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'train', help='deca mode')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
