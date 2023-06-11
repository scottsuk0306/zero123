
import numpy as np
import voxnerf.vox as vox
import torch

model_path = "zero123/3drec/experiments/exp_wild/scene-pikachu-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-10000.0_near-view-wt-10000.0/ckpt/step_10000.pt"
model = vox.VoxRF()
model.load_state_dict(torch.load(model_path))
model.eval()