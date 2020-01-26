import cross_corr
import torch
import torch.nn.functional as F
import cv2
import ipdb
from scipy.misc import imsave
st = ipdb.set_trace
img = torch.tensor(cv2.imread('new.jpg')/255.0).permute(2,0,1).unsqueeze(0).cuda()
img = F.interpolate(img,[100,100]).to(torch.float32)
mbr = cross_corr.meshgrid_based_rotation(100,100,100, angleIncrement=10.0)
# rotated_inputs = mbr.rotate2D(img,interpolation="nearestNeighbor")
rotated_inputs = mbr.rotate2D(img,interpolation="bilinear")
# st()


rotated_inputs = rotated_inputs.permute(0,2,1,3,4)
for index in range(36):
	rotated_inputs_i = rotated_inputs[0,index]
	rotated_inputs_i_np = rotated_inputs_i.permute(1,2,0).cpu().numpy()
	imsave(f"rotated/{index}.png",rotated_inputs_i_np)
# rotated_inputs_flatten = rotated_inputs.view(-1, rotated_inputs.shape[1], embedding_size)
# rot_inputs_sqr = torch.sum(rotated_inputs_flatten ** 2, dim=2, keepdim=True)
# rot_distances = torch.addmm(codebook_sqr + rot_inputs_sqr, rotated_inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
# dB, dA, dF = rot_distances.shape
# _, rot_indices_flatten = torch.min(rot_distances.view(dB, -1))
# rot_indices = rot_indices_flatten%dF
# rot_indices = rot_indices.view(*inputs_size[:-1])