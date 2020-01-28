import cross_corr
import torch
import torch.nn.functional as F
import cv2
import ipdb
from scipy.misc import imsave
st = ipdb.set_trace
img = torch.tensor(cv2.imread('new.jpg')/255.0).permute(2,0,1).unsqueeze(0).cuda()
img_size = 100
img = F.interpolate(img,[img_size,img_size]).to(torch.float32)
orig = img.squeeze(0).permute(1,2,0).cpu().numpy()
imsave(f"rotated/original.png",orig)


mbr = cross_corr.meshgrid_based_rotation(img_size,img_size,img_size, angleIncrement=10.0)
rotated_inputs = mbr.rotate2D(img,interpolation="bilinear")


rotated_inputs = rotated_inputs.permute(0,2,1,3,4)
# for index in range(36):
# 	rotated_inputs_i = rotated_inputs[0,index]
# 	rotated_inputs_i_np = rotated_inputs_i.permute(1,2,0).cpu().numpy()
# 	imsave(f"rotated/{index}.png",rotated_inputs_i_np)


B,angles,C,H,W = list(rotated_inputs.shape)
C = C*H*W
# assert(C==embedding_size)
rot_input = rotated_inputs.reshape(B,angles,-1)
index_to_take = 5
codebook = rot_input[0,index_to_take:index_to_take+1] 
codebook_randn = torch.randn(3,codebook.shape[1]).cuda()
codebook = torch.cat([codebook,codebook_randn],dim=0)
codebook_sqr = torch.sum(codebook ** 2, dim=1)
rot_inputs_sqr = torch.sum(rot_input ** 2, dim=2, keepdim=True)
rot_distances = (rot_inputs_sqr + codebook_sqr - 2 * torch.matmul(rot_input, codebook.t()))                    

# st()
dB, dA, dF = rot_distances.shape
rot_distances = rot_distances.view(B, -1)
rotIdxMin = torch.argmin(rot_distances, dim=1).unsqueeze(1)
# st()
best_rotations = rotIdxMin//dF # Find the rotation for min distance
best_rotations = best_rotations.squeeze(1)
encoding_indices = rotIdxMin%dF # Find the best index (which will be column in rotAngle-index grid)
encoding_indices = encoding_indices.squeeze(1)
indices_flatten = encoding_indices.view(-1)
codes_flatten = torch.index_select(codebook, dim=0,index=indices_flatten)
codes = codes_flatten.view_as(img)

codes_i = codes[0]
codes_i = codes_i.permute(1,2,0).cpu().numpy()
imsave(f"rotated/most_similar.png",codes_i)

codes_rotated = mbr.rotate2D_pose(codes,best_rotations).squeeze(2)
codes_rotated_i = codes_rotated[0]
codes_rotated_i_np = codes_rotated_i.permute(1,2,0).cpu().numpy()
imsave(f"rotated/unrotated.png",codes_rotated_i_np)
# st()
print("check")