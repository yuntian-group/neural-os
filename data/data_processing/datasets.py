import pandas as pd               # For reading CSV files with image and MRI paths.
import numpy as np                # For handling numerical operations, arrays, and data types.
from PIL import Image    
import PIL         # To open, manipulate, and process images.
import torchvision.transforms as transforms  # For performing image augmentations like random horizontal flips.
from torch.utils.data import Dataset  # Dataset class to inherit for custom datasets.
import os
import torch
from typing import List
from latent_diffusion.ldm.modules.encoders.modules import BERTTokenizer
import ast

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Personalize0(Dataset):
    """
    class dataset for imagenet.
    """
    def __init__(self,
                 data_csv_path,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 val=False
                 ):
        self.data_path = data_csv_path
        
        data = pd.read_csv(data_csv_path)
        self.image_paths = data["Image_path"]
        self.labels = data['Label']
    
        if val:
            self.image_paths=self.image_paths[:1024]
            self.labels=self.lables[:1024]

        self._length = len(self.image_paths)
        self.size = size
        # self.interpolation = {"linear": PIL.Image.LINEAR,
        #                       "bilinear": PIL.Image.BILINEAR,
        #                       "bicubic": PIL.Image.BICUBIC,
        #                       "lanczos": PIL.Image.LANCZOS,
        #                       }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        image = Image.open(self.image_paths[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        # img = np.array(image).astype(np.uint8)
        # crop = min(img.shape[0], img.shape[1])
        # h, w, = img.shape[0], img.shape[1]
        # img = img[(h - crop) // 2:(h + crop) // 2,
        #       (w - crop) // 2:(w + crop) // 2]

        # image = Image.fromarray(img)
        # if self.size is not None:
        #     image = image.resize((self.size, self.size), resample=self.interpolation)

        # image = self.flip(image)
        # image = np.array(image).astype(np.uint8)

        images, labels = get_all_images(self.image_paths, self.labels)

        image = np.array(image).astype(np.float32)

        device = 'cpu'

        # example["image"] = images.to(device).repeat(5,1,1,1,1) 
        # example["label"] = labels.to(device).repeat(5,1) 

        example["image"] = torch.tensor((image / 127.5 - 1.0).astype(np.float32), device=device).unsqueeze(0).unsqueeze(0).repeat(800,1,1,1,1) # n b w h c
        example["class_label"] = torch.tensor(self.labels[i], device=device).unsqueeze(0).unsqueeze(0).repeat(800,1)

        return example
    
def get_all_images(images, labels):

    t_images = []
    t_labels = []

    for path, label in zip(images, labels):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.float32)

        image = torch.tensor((image / 127.5 - 1.0).astype(np.float32)).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)

        t_images.append(image)
        t_labels.append(label)

    return torch.stack(t_images), torch.stack(t_labels)






class ActionsData(Dataset):
    """
    class dataset for csllm. includes images and corresponding action labels.
    """
    def __init__(self,
                 data_csv_path,
                 size=256,
                 val=False
                 ):
        self.data_path = data_csv_path

        bert_tokenizer = BERTTokenizer(vq_interface=False)
        
        data = pd.read_csv(data_csv_path)
        self.image_paths = data["Image_path"]

        # data["Label"] = data['Label'].apply(lambda label: bert_tokenizer.encode(label)) #tokenizes the actions.
        self.labels = data['Label']
    
        if val:
            self.image_paths=self.image_paths[:1024]
            self.labels=self.lables[:1024]

        self._length = len(self.image_paths) * 10
        self.size = size


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        # i = 157
        # i = i % 180
        i = 50 if i%2 == 0 else 0
        image = Image.open(self.image_paths[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.float32)

        device = 'cpu'

        #single sample overfit
        example["image"] = torch.tensor((image / 127.5 - 1.0).astype(np.float32))# n b w h c
        example["caption"] = str(self.labels[i])

        # example["image"] = torch.tensor((image / 127.5 - 1.0).astype(np.float32))# n b w h c
        # example["caption"] = self.labels[i]

        return example   

class ActionsSequenceData(Dataset):
    """
    class dataset for csllm. includes image sequences and corresponding action sequences for cond.
    """
    def __init__(self,
                 data_csv_path,
                 size=256,
                 val=False
                 ):
        self.data_path = data_csv_path

        bert_tokenizer = BERTTokenizer(vq_interface=False)
        
        data = pd.read_csv(data_csv_path)
        self.image_seq_paths = data["Image_seq_cond_path"].apply(ast.literal_eval).to_list()
        self.actions_seq = data['Action_seq'].apply(ast.literal_eval).to_list()
        self.targets = data['Target_image'].to_list()

        # if val:
        #     self.image_paths=self.image_seq_paths[:1024]
        #     self.labels=self.image_seq_paths[:1024]

        self._length = len(self.image_seq_paths)
        self.size = size


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        """
        takes a sequence of cond. images and actions and a single target.
        """
        example = dict()
        i = i % 173
        # i = 0 if i % 2 == 0 else 50


        #single sample overfit
        example["image"] = self.process_image(self.targets[i]) # torch.stack(image_target) # n b w h c
        example["caption"] = ' '.join(self.actions_seq[i]) # actions_cond #untokenized actions
        example['c_concat'] = torch.stack([self.process_image(image_path) for image_path in self.image_seq_paths[i]]) # sequence of images

        return example 

    def process_image(self, image_path):  

        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = (np.array(image) / 127.5 - 1.0).astype(np.float32)

        return torch.tensor(image)


        

        

#class label testing stuff
class PersonalizeTrain0(Personalize0):
    def __init__(self, **kwargs):
        super().__init__(data_csv_path='/u4/jlrivard/latent-diffusion/data/train_256x256/train_info.csv')

class PersonalizeVal0(Personalize0):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file='' ,val=True,
                         flip_p=flip_p)




class CsllmTrain(ActionsData):
    def __init__(self, **kwargs):
        super().__init__(data_csv_path='/u4/jlrivard/latent-diffusion/data/train_256x256_actions/train_info.csv')

class PersonalizeVal0(Personalize0):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file='' ,val=True,
                         flip_p=flip_p)
        
class CsllmTrainSeq(ActionsSequenceData):
    def __init__(self, **kwargs):
        super().__init__(data_csv_path='/u4/jlrivard/latent-diffusion/data/train_256x256_w_actions_binned/train_sequence_info.csv')


# 'C:/Users/Luke/latent-diffusion/data/train_256x256_w_actions_seq_2/train_sequence_info.csv'