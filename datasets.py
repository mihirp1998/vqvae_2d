import os
import csv
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import ipdb
import numpy as np
import torch
import pickle
from scipy.misc import imsave
import cv2

st = ipdb.set_trace
hyp_N = 3
hyp_B = 1

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Borrowed from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MiniImagenet(data.Dataset):

    base_folder = '/data/lisa/data/miniimagenet'
    filename = 'miniimagenet.zip'
    splits = {
        'train': 'train.csv',
        'valid': 'val.csv',
        'test': 'test.csv'
    }

    def __init__(self, root, train=False, valid=False, test=False,
                 transform=None, target_transform=None, download=False):
        super(MiniImagenet, self).__init__()
        self.root = root
        self.train = train
        self.valid = valid
        self.test = test
        self.transform = transform
        self.target_transform = target_transform

        if not (((train ^ valid ^ test) ^ (train & valid & test))):
            raise ValueError('One and only one of `train`, `valid` or `test` '
                'must be True (train={0}, valid={1}, test={2}).'.format(train,
                valid, test))

        self.image_folder = os.path.join(os.path.expanduser(root), 'images')
        if train:
            split = self.splits['train']
        elif valid:
            split = self.splits['valid']
        elif test:
            split = self.splits['test']
        else:
            raise ValueError('Unknown split.')
        self.split_filename = os.path.join(os.path.expanduser(root), split)
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use `download=True` '
                               'to download it')

        # Extract filenames and labels
        self._data = []
        with open(self.split_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip the header
            for line in reader:
                self._data.append(tuple(line))
        self._fit_label_encoding()

    def __getitem__(self, index):
        filename, label = self._data[index]
        image = pil_loader(os.path.join(self.image_folder, filename))
        label = self._label_encoder[label]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def _fit_label_encoding(self):
        _, labels = zip(*self._data)
        unique_labels = set(labels)
        self._label_encoder = dict((label, idx)
            for (idx, label) in enumerate(unique_labels))

    def _check_exists(self):
        return (os.path.exists(self.image_folder) 
            and os.path.exists(self.split_filename))

    def download(self):
        from shutil import copyfile
        from zipfile import ZipFile

        # If the image folder already exists, break
        if self._check_exists():
            return True

        # Create folder if it does not exist
        root = os.path.expanduser(self.root)
        if not os.path.exists(root):
            os.makedirs(root)

        # Copy the file to root
        path_source = os.path.join(self.base_folder, self.filename)
        path_dest = os.path.join(root, self.filename)
        print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
        copyfile(path_source, path_dest)

        # Extract the dataset
        print('Extract files from `{0}`...'.format(path_dest))
        with ZipFile(path_dest, 'r') as f:
            f.extractall(root)

        # Copy CSV files
        for split in self.splits:
            path_source = os.path.join(self.base_folder, self.splits[split])
            path_dest = os.path.join(root, self.splits[split])
            print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
            copyfile(path_source, path_dest)
        print('Done!')

    def __len__(self):
        return len(self._data)



class Clevr(data.Dataset):
    def __init__(self, root, mod="temp",object_level=False, train=False, valid=False, test=False,
                 transform=None, target_transform=None, download=False):
        super(Clevr, self).__init__()
        self.root = root
        self.mod_folder = f"{root}/npys"
        self.transform = transform
        self.target_transform = target_transform
        self.image_folder = os.path.join(os.path.expanduser(root), 'images')
        self._data = []
        self.object_level = object_level
        
        if train:
            modfile = f"{self.mod_folder}/{mod}t.txt"
        elif valid:
            modfile = f"{self.mod_folder}/{mod}v.txt"
        elif test:
            modfile = f"{self.mod_folder}/{mod}v.txt"
        else:
            raise ValueError('Unknown split.')

        self._data = []
        # Extract filenames and labels
        with open(modfile, 'r') as f:
            for filename_i in f.readlines():
                filename = filename_i[:-1]
                filepath = f"{self.mod_folder}/{filename}"
                self._data.append(filepath)
        # self._fit_label_encoding()
    def bbox_rearrange(self,tree,boxes= [],classes=[]):
        for i in range(0, tree.num_children):
            updated_tree,boxes,classes = self.bbox_rearrange(tree.children[i],boxes=boxes,classes=classes)
            tree.children[i] = updated_tree     
        if tree.function == "describe":
            xmax,ymax,zmin,xmin,ymin,zmax = tree.bbox_origin
            box = np.array([xmin,ymin,zmin,xmax,ymax,zmax])
            tree.bbox_origin = box
            boxes.append(box)
            classes.append(tree.word)
        return tree,boxes,classes

    def trees_rearrange(self,tree):
        tree,boxes,classes = self.bbox_rearrange(tree,boxes=[],classes=[])
        # st()
        boxes = np.stack(boxes)
        classes = np.stack(classes)
        N,_  = boxes.shape 
        scores = np.pad(np.ones([N]),[0,hyp_N-N])
        boxes = np.pad(boxes,[[0,hyp_N-N],[0,0]])
        # st()
        classes = np.pad(classes,[0,hyp_N-N])
        scores = np.expand_dims(scores,axis=0)
        boxes = np.expand_dims(boxes,axis=0)
        classes = np.expand_dims(classes,axis=0)
        return boxes,scores,classes

    def get_alignedboxes2thetaformat(self,aligned_boxes):
        aligned_boxes = torch.reshape(aligned_boxes,[hyp_B,hyp_N,6])
        B,N,_ = list(aligned_boxes.shape)
        xmin,ymin,zmin,xmax,ymax,zmax = torch.unbind(torch.tensor(aligned_boxes), dim=-1)
        xc = (xmin+xmax)/2.0
        yc = (ymin+ymax)/2.0
        zc = (zmin+zmax)/2.0
        w = xmax-xmin
        h = ymax - ymin
        d = zmax - zmin
        zeros = torch.zeros([B,N])
        boxes = torch.stack([xc,yc,zc,w,h,d,zeros,zeros,zeros],dim=-1)
        return boxes


    def eye_3x3(self,B):
        rt = torch.eye(3, device=torch.device('cpu')).view(1,3,3).repeat([B, 1, 1])
        return rt

    def eye_4x4(self,B):
        rt = torch.eye(4, device=torch.device('cpu')).view(1,4,4).repeat([B, 1, 1])
        return rt        

    def merge_rt(self,r, t):
        # r is B x 3 x 3
        # t is B x 3
        B, C, D = list(r.shape)
        B2, D2 = list(t.shape)
        assert(C==3)
        assert(D==3)
        assert(B==B2)
        assert(D2==3)
        t = t.view(B, 3)
        rt = self.eye_4x4(B)
        rt[:,:3,:3] = r
        rt[:,:3,3] = t
        return rt

    def eul2rotm(self,rx, ry, rz):
        # inputs are shaped B
        # this func is copied from matlab
        # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
        #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
        #        -sy            cy*sx             cy*cx]
        rx = torch.unsqueeze(rx, dim=1)
        ry = torch.unsqueeze(ry, dim=1)
        rz = torch.unsqueeze(rz, dim=1)
        # these are B x 1
        sinz = torch.sin(rz)
        siny = torch.sin(ry)
        sinx = torch.sin(rx)
        cosz = torch.cos(rz)
        cosy = torch.cos(ry)
        cosx = torch.cos(rx)
        r11 = cosy*cosz
        r12 = sinx*siny*cosz - cosx*sinz
        r13 = cosx*siny*cosz + sinx*sinz
        r21 = cosy*sinz
        r22 = sinx*siny*sinz + cosx*cosz
        r23 = cosx*siny*sinz - sinx*cosz
        r31 = -siny
        r32 = sinx*cosy
        r33 = cosx*cosy
        r1 = torch.stack([r11,r12,r13],dim=2)
        r2 = torch.stack([r21,r22,r23],dim=2)
        r3 = torch.stack([r31,r32,r33],dim=2)
        r = torch.cat([r1,r2,r3],dim=1)
        return r

    def matmul2(self,mat1, mat2):
        return torch.matmul(mat1, mat2)



    def convert_box_to_ref_T_obj(self,box3D):
        # turn the box into obj_T_ref (i.e., obj_T_cam)
        B = list(box3D.shape)[0]
        
        # box3D is B x 9
        x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box3D, axis=1)
        rot0 = self.eye_3x3(B)
        tra = torch.stack([x, y, z], axis=1)
        center_T_ref = self.merge_rt(rot0, -tra)
        # center_T_ref is B x 4 x 4
        
        t0 = torch.zeros([B, 3])
        rot = self.eul2rotm(rx, -ry, -rz)
        obj_T_center = self.merge_rt(rot, t0)
        # this is B x 4 x 4

        # we want obj_T_ref
        # first we to translate to center,
        # and then rotate around the origin
        obj_T_ref = self.matmul2(obj_T_center, center_T_ref)
        # return the inverse of this, so that we can transform obj corners into cam coords
        ref_T_obj = obj_T_ref.inverse()
        return ref_T_obj        

    def apply_4x4(self,RT, xyz):
        B, N, _ = list(xyz.shape)
        ones = torch.ones_like(xyz[:,:,0:1])
        xyz1 = torch.cat([xyz, ones], 2)
        xyz1_t = torch.transpose(xyz1, 1, 2)
        # this is B x 4 x N
        xyz2_t = torch.matmul(RT, xyz1_t)
        xyz2 = torch.transpose(xyz2_t, 1, 2)
        xyz2 = xyz2[:,:,:3]
        return xyz2


    def transform_boxes_to_corners_single(self,boxes):
        N, D = list(boxes.shape)
        assert(D==9)
        
        xc,yc,zc,lx,ly,lz,rx,ry,rz = torch.unbind(boxes, axis=1)
        # these are each shaped N

        ref_T_obj = self.convert_box_to_ref_T_obj(boxes)

        xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
        ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
        zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
        
        xyz_obj = torch.stack([xs, ys, zs], axis=2)
        # centered_box is N x 8 x 3

        xyz_ref = self.apply_4x4(ref_T_obj, xyz_obj)
        # xyz_ref is N x 8 x 3
        return xyz_ref


    def transform_boxes_to_corners(self,boxes):
        # returns corners, shaped B x N x 8 x 3
        B, N, D = list(boxes.shape)
        assert(D==9)
        
        __p = lambda x: self.pack_seqdim(x, B)
        __u = lambda x: self.unpack_seqdim(x, B)

        boxes_ = __p(boxes)
        corners_ = self.transform_boxes_to_corners_single(boxes_)
        corners = __u(corners_)
        return corners


    def pack_seqdim(self,tensor, B):
        shapelist = list(tensor.shape)
        B_, S = shapelist[:2]
        assert(B==B_)
        otherdims = shapelist[2:]
        tensor = torch.reshape(tensor, [B*S]+otherdims)
        return tensor


    def unpack_seqdim(self,tensor, B):
        shapelist = list(tensor.shape)
        BS = shapelist[0]
        assert(BS%B==0)
        otherdims = shapelist[1:]
        S = int(BS/B)
        tensor = torch.reshape(tensor, [B,S]+otherdims)
        return tensor

    def safe_inverse(self,a): #parallel version
        B, _, _ = list(a.shape)
        inv = a.clone()
        r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

        inv[:, :3, :3] = r_transpose
        inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

        return inv


    def pack_boxdim(self,tensor, N):
        shapelist = list(tensor.shape)
        B, N_, C = shapelist[:3]
        assert(N==N_)
        # assert(C==8)
        otherdims = shapelist[3:]
        tensor = torch.reshape(tensor, [B,N*C]+otherdims)
        return tensor

    def unpack_boxdim(self,tensor, N):
        shapelist = list(tensor.shape)
        B,NS = shapelist[:2]
        assert(NS%N==0)
        otherdims = shapelist[2:]
        S = int(NS/N)
        tensor = torch.reshape(tensor, [B,N,S]+otherdims)
        return tensor

    def split_intrinsics(self,K):
        # K is B x 3 x 3 or B x 4 x 4
        fx = K[:,0,0]
        fy = K[:,1,1]
        x0 = K[:,0,2]
        y0 = K[:,1,2]
        return fx, fy, x0, y0

    def apply_pix_T_cam(self,pix_T_cam, xyz):

        fx, fy, x0, y0 = self.split_intrinsics(pix_T_cam)
        
        # xyz is shaped B x H*W x 3
        # returns xy, shaped B x H*W x 2
        
        B, N, C = list(xyz.shape)
        assert(C==3)
        
        x, y, z = torch.unbind(xyz, axis=-1)

        fx = torch.reshape(fx, [B, 1])
        fy = torch.reshape(fy, [B, 1])
        x0 = torch.reshape(x0, [B, 1])
        y0 = torch.reshape(y0, [B, 1])

        EPS=1e-6
        x = (x*fx)/(z+EPS)+x0
        y = (y*fy)/(z+EPS)+y0
        xy = torch.stack([x, y], axis=-1)
        return xy

    def get_ends_of_corner(self,boxes):
        min_box = torch.min(boxes,dim=2,keepdim=True).values
        max_box = torch.max(boxes,dim=2,keepdim=True).values
        boxes_ends = torch.cat([min_box,max_box],dim=2)
        return boxes_ends


    def __getitem__(self, index):
        data = pickle.load(open(self._data[index],"rb"))
        view_to_take = np.random.randint(0,24)
        random_rgb = data['rgb_camXs_raw'][view_to_take][...,:3]
        camR_T_origin_raw = torch.from_numpy(data["camR_T_origin_raw"][view_to_take:view_to_take+1]).unsqueeze(dim=0)
        pix_T_cams_raw = torch.from_numpy(data["pix_T_cams_raw"][view_to_take:view_to_take+1]).unsqueeze(dim=0)
        origin_T_camXs_raw = torch.from_numpy(data["origin_T_camXs_raw"][view_to_take:view_to_take+1]).unsqueeze(dim=0)
        if self.object_level:
            __pb = lambda x: self.pack_boxdim(x, hyp_N)
            __ub = lambda x: self.unpack_boxdim(x, hyp_N)            
            __p = lambda x: self.pack_seqdim(x, hyp_B)
            __u = lambda x: self.unpack_seqdim(x, hyp_B)

            pix_T_cams = pix_T_cams_raw
            # cam_T_velos = feed["cam_T_velos"]
            origin_T_camRs = __u(self.safe_inverse(__p(camR_T_origin_raw)))
            origin_T_camXs = origin_T_camXs_raw


            camRs_T_camXs = __u(torch.matmul(self.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
            # st()
            camXs_T_camRs = __u(self.safe_inverse(__p(camRs_T_camXs)))
            camX0_T_camRs = camXs_T_camRs[:,0]
            camR_T_camX0  = self.safe_inverse(camX0_T_camRs)

            tree_file_name = data['tree_seq_filename']
            tree_file_path = f"{self.root}/{tree_file_name}"
            trees = pickle.load(open(tree_file_path,"rb"))
            gt_boxesR,scores,classes = self.trees_rearrange(trees)
            gt_boxesR = torch.from_numpy(gt_boxesR)
            gt_boxesR_end = torch.reshape(gt_boxesR,[hyp_B,hyp_N,2,3])

            gt_boxesR_theta = self.get_alignedboxes2thetaformat(gt_boxesR_end)
            gt_boxesR_corners = self.transform_boxes_to_corners(gt_boxesR_theta)

            gt_boxesX0_corners = __ub(self.apply_4x4(camX0_T_camRs, __pb(gt_boxesR_corners)))
            gt_cornersX0_pix = __ub(self.apply_pix_T_cam(pix_T_cams[:,0], __pb(gt_boxesX0_corners)))
            boxes_pix = self.get_ends_of_corner(gt_cornersX0_pix)
            boxes_pix = torch.clamp(boxes_pix,min=0)            
            vis = True
            tmp_box = boxes_pix[0,0]
            lower,upper = torch.unbind(tmp_box)
            xmin,ymin = torch.floor(lower).to(torch.int16)
            xmax,ymax = torch.ceil(upper).to(torch.int16)
            object_rgb = random_rgb[ymin:ymax,xmin:xmax]
            random_rgb = cv2.resize(object_rgb,(int(64),int(64)))
        # label = self._label_encoder[label]
        # tree_file = data['tree_seq_filename']
        # tree_file_path = f""
        # st()
        if self.transform is not None:
            image = self.transform(random_rgb)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        label = 0
        # st()
        return image, label

    def _fit_label_encoding(self):
        _, labels = zip(*self._data)
        unique_labels = set(labels)
        self._label_encoder = dict((label, idx)
            for (idx, label) in enumerate(unique_labels))

    def _check_exists(self):
        return (os.path.exists(self.image_folder) 
            and os.path.exists(self.split_filename))

    def download(self):
        from shutil import copyfile
        from zipfile import ZipFile

        # If the image folder already exists, break
        if self._check_exists():
            return True

        # Create folder if it does not exist
        root = os.path.expanduser(self.root)
        if not os.path.exists(root):
            os.makedirs(root)

        # Copy the file to root
        path_source = os.path.join(self.base_folder, self.filename)
        path_dest = os.path.join(root, self.filename)
        print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
        copyfile(path_source, path_dest)

        # Extract the dataset
        print('Extract files from `{0}`...'.format(path_dest))
        with ZipFile(path_dest, 'r') as f:
            f.extractall(root)

        # Copy CSV files
        for split in self.splits:
            path_source = os.path.join(self.base_folder, self.splits[split])
            path_dest = os.path.join(root, self.splits[split])
            print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
            copyfile(path_source, path_dest)
        print('Done!')

    def __len__(self):
        return len(self._data)

if __name__ == "__main__":
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(128),        
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    import socket   
    if "compute" in socket.gethostname():
        root_dataset = "/projects/katefgroup/datasets/clevr_veggies/"
    else:
        root_dataset = "/media/mihir/dataset/clevr_veggies/"
    clevr = Clevr(root_dataset,mod="cg",train=True,transform=transform,object_level= True)
    fixed_images, _ = next(iter(clevr))
    print("hello")