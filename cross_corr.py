import numpy as np 
import torch
import kornia
import ipdb 
st = ipdb.set_trace

class meshgrid_based_rotation:
    """
    This helper precomputed a fixed grid of indices for rotation.
    This is suitable when you always want to rotate a fixed list of angles for all
    the voxels in the batch and you don't want to compute the rotation matrix
    everytimes you want to transform
    """
    def __init__(self, D, H, W, angleIncrement=10):
        self.D = D
        self.H = H
        self.W = W
        self.centerD = self.D//2
        self.centerW = self.W//2
        self.EPS = 1e-5
        self.device = torch.device("cuda")
        self.angleIncrement = angleIncrement
        # with torch.no_grad():
        self.anglesDeg = -1*torch.arange(0, 360, angleIncrement).to(self.device)
        self.anglesRad = kornia.deg2rad(self.anglesDeg).to(self.device)
        self.precomputeMeshGrids()
    
    '''
    Get rotation matrix which rotates embedding along center
    https://math.stackexchange.com/questions/2093314/rotation-matrix-of-rotation-around-a-point-other-than-the-origin
    '''
    def precomputeMeshGrids(self):
        # Meshgrid along D and W
        
        dInd = torch.arange(self.D).to(self.device)
        wInd = torch.arange(self.W).to(self.device)

        dMesh, wMesh = torch.meshgrid(dInd, wInd)

        cosThetas = torch.cos(self.anglesRad)
        sinThetas = torch.sin(self.anglesRad)
        numAngles = self.anglesRad.shape[0]
        self.numAngles = numAngles
        
        dMesh = dMesh.unsqueeze(0).repeat(numAngles,1,1)
        wMesh = wMesh.unsqueeze(0).repeat(numAngles,1,1)

        cosThetas = cosThetas.view(-1, 1, 1)
        sinThetas = sinThetas.view(-1, 1, 1)

        # We will be rotating along the center.
        self.dRot = cosThetas*dMesh - sinThetas*wMesh - cosThetas*self.centerD + sinThetas*self.centerW + self.centerD #+ self.EPS # [36, 5, 5]
        self.wRot   = sinThetas*dMesh + cosThetas*wMesh - sinThetas*self.centerD - cosThetas*self.centerW + self.centerW #+ self.EPS # [36, 5, 5]
        
        self.dRot = torch.clamp(self.dRot, 0+self.EPS, self.D-1-self.EPS)
        self.wRot = torch.clamp(self.wRot, 0+self.EPS, self.W-1-self.EPS)
    
    def rotateTensor(self, tensor, interpolation="bilinear"):
        assert tensor.ndim == 5, "Tensor should have 5 dimensions (B,C,D,H,W)"

        B,C,D,H,W = tensor.shape
        tensor = tensor.permute(0, 1, 3, 2, 4) # torch.Size([2, 32, 16, 16, 16])
        tensor = tensor.reshape(B, C*H, D, W)
        rotated =  self.rotate2D(tensor, interpolation) # torch.Size([2, 512, 36, 16, 16])

        rotated = rotated.reshape(B, C, H, self.numAngles, D, W) 
        rotated = rotated.permute(0, 3, 1, 4, 2, 5) # B, numAngles, C, D, H, W
        return rotated
    
    def rotate2D(self, tensor, interpolation="bilinear"):
        if interpolation == "nearestNeighbor":
            out = self.nearestNeighborInterpolation(tensor)
        else:
            out = self.bilinearInterpolation(tensor)
            out[:,:,0,:,:] = tensor # 0 degree rotation is original tensor.
            out[:, :, :, self.centerD, self.centerW] = tensor.unsqueeze(2)[:, :, :, self.centerD, self.centerW] # set the value of center pixel
        return out

    
    def bilinearInterpolation(self, tensor):
        dfloor, dceil, wfloor, wceil = self.getFloorAndCeil() # torch.Size([36, 60, 60])
        fq12 = tensor[:,:,dceil, wfloor] # torch.Size([2, 3, 36, 60, 60])
        fq22 = tensor[:,:,dceil, wceil]
        fq11 = tensor[:,:,dfloor,wfloor]
        fq21 = tensor[:,:,dfloor,wceil]
        y1, y2, x1, x2 = dfloor.unsqueeze(0).unsqueeze(0), dceil.unsqueeze(0).unsqueeze(0), wfloor.unsqueeze(0).unsqueeze(0), wceil.unsqueeze(0).unsqueeze(0)
        y = self.dRot.unsqueeze(0).unsqueeze(0)
        x = self.wRot.unsqueeze(0).unsqueeze(0)
        numerator = fq11*(x2-x)*(y2-y) + fq21*(x-x1)*(y2-y) + fq12*(x2-x)*(y-y1) + fq22*(x-x1)*(y-y1)
        denominator = (x2-x1)*(y2-y1) + self.EPS
        out = numerator/denominator
        return out


    def nearestNeighborInterpolation(self, tensor):
        
        dfloor, dceil, wfloor, wceil = self.getFloorAndCeil()
        out = tensor[:, :, dfloor, wfloor]
        return out

    def getFloorAndCeil(self):
        dfloor = torch.floor(self.dRot).long()
        dceil = torch.ceil(self.dRot).long()
        wfloor = torch.floor(self.wRot).long()
        wceil = torch.ceil(self.wRot).long()
        return dfloor, dceil, wfloor, wceil
    