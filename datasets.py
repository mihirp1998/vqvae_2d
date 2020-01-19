import os
import csv
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import ipdb
import numpy as np
import pickle
st = ipdb.set_trace

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
    def __init__(self, root, mod="temp", train=False, valid=False, test=False,
                 transform=None, target_transform=None, download=False):
        super(Clevr, self).__init__()
        self.root = root
        self.mod_folder = f"{root}/npys"
        self.transform = transform
        self.target_transform = target_transform
        self.image_folder = os.path.join(os.path.expanduser(root), 'images')
        self._data = []
        
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

    def trees_rearrange(self,tree):
        tree,boxes,classes = bbox_rearrange(tree,boxes=[],classes=[])
        # st()
        boxes = np.stack(boxes)
        classes = np.stack(classes)
        N,_  = boxes.shape 
        scores = np.pad(np.ones([N]),[0,hyp.N-N])
        boxes = np.pad(boxes,[[0,hyp.N-N],[0,0]])
        # st()
        classes = np.pad(classes,[0,hyp.N-N])
        # boxes.append(boxes[-1])
        return boxes,scores,classes

    def __getitem__(self, index):
        data = pickle.load(open(self._data[index],"rb"))
        view_to_take = np.random.randint(0,24)
        random_rgb = data['rgb_camXs_raw'][view_to_take][...,:3]
        tree_file_name = data['tree_seq_filename']
        tree_file_path = f"{self.root}/tree_file_name"
        trees = pickle.load(open(tree_file_path,"rb"))
        gt_boxesR,scores,classes = nlu.trees_rearrange(trees)
        st()

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
    clevr = Clevr("/media/mihir/dataset/clevr_veggies/",mod="cg",train=True,transform=transform)
    fixed_images, _ = next(iter(clevr))
    print("hello")