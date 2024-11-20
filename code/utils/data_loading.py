# #数据集加载实验
# import os
# import random
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# import albumentations as A
# import albumentations.pytorch

# class Compose(A.Compose):
#     def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
#         super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)
    
#     def __call__(self, image):
#         augmented = super().__call__(image=np.array(image))
#         return augmented['image']
    
# class MAMLDataset(Dataset):
#     def __init__(self, root_dir, img_size, trn, kshot=1, k_qry=1, val_class=None):
#         self.root_dir = root_dir
#         self.trn = trn
#         self.q = k_qry
#         if self.trn:
#             self.classes = sorted([folder for folder in os.listdir(os.path.join(root_dir, 'trn')) if not folder.startswith('.')])
#         else:
#             self.classes = [val_class]
#         self.img_metadata = self.build_img_metadata()
#         self.dirs = [os.path.join(os.path.join(root_dir,'trn'), class_name) for class_name in self.classes]
#         self.kshot=kshot
#         self.transform = Compose([
#             A.Resize(img_size, img_size),
#             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             A.pytorch.transforms.ToTensorV2(),
#         ])
        
#     def __len__(self):
#         if self.trn:
#             return len(self.img_metadata)// 3 // self.q
#         else:
#             return len(self.img_metadata)

#     def __getitem__(self, idx):
#         if self.trn:
#             q_images = []
#             q_masks = []
#             s_images = []
#             s_masks = []
#             for dir in self.dirs:
#                 image_dir= os.path.join(dir, 'imgs')
#                 q_mask_dir = os.path.join(dir, 'masks')
#                 files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
#                 q_selected_files = files[idx*self.q:idx*self.q+self.q]
#                 q_images_class = []
#                 q_mask_class = []
#                 for file in q_selected_files:
#                     q_image = Image.open(os.path.join(image_dir, file)).convert('RGB')
#                     q_image = self.transform(q_image)
#                     q_images_class.append(q_image)
#                     q_mask = self.read_mask(os.path.join(q_mask_dir, os.path.splitext(file)[0]+".png"))
#                     q_mask = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0).float(), q_image.size()[-2:], mode='nearest').squeeze()
#                     q_mask_class.append(q_mask)
#                 q_images.append(torch.stack(q_images_class))
#                 q_masks.append(torch.stack(q_mask_class))
#                 s_selected_files=[]
#                 s_images_class=[]
#                 s_mask_class=[]
#                 while True:  
#                     support_file = random.choice(files)
#                     if support_file not in q_selected_files: s_selected_files.append(support_file)
#                     if len(s_selected_files) == self.kshot: break

#                 for file in s_selected_files:
#                     s_image = Image.open(os.path.join(image_dir, file)).convert('RGB')
#                     s_image = self.transform(s_image)
#                     s_images_class.append(s_image)
#                     s_mask = self.read_mask(os.path.join(q_mask_dir, os.path.splitext(file)[0]+".png"))
#                     s_mask = F.interpolate(s_mask.unsqueeze(0).unsqueeze(0).float(), s_image.size()[-2:], mode='nearest').squeeze()
#                     s_mask_class.append(s_mask)
#                 s_images.append(torch.stack(s_images_class))
#                 s_masks.append(torch.stack(s_mask_class))
#             query_img = torch.stack(q_images)
#             query_mask = torch.stack(q_masks)
#             support_img = torch.stack(s_images)
#             support_mask = torch.stack(s_masks)
#         else: # VAL
#             query_name, support_name, class_name = self.sample_episode(idx)
#             query_img, query_mask, support_img, support_mask = self.load_frame(query_name, support_name)
#             query_img = self.transform(query_img)
#             query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

#             support_img = self.transform(support_img)
#             support_mask = F.interpolate(support_mask.unsqueeze(0).unsqueeze(0).float(), support_img.size()[-2:], mode='nearest').squeeze()
        
#         batch = {'query_img': query_img,
#                  'query_mask': query_mask,
#                  'support_img': support_img,
#                  'support_mask': support_mask,
#                  }
#         return batch
#     def load_frame(self, query_name, support_name):
#         query_img = Image.open(query_name).convert('RGB')
#         support_img = Image.open(support_name).convert('RGB')
#         # support_imgs = [Image.open(name).convert('RGB') for name in support_names]
#         query_id = query_name.split('/')[-1].split('.')[0]
#         query_name = os.path.join(os.path.dirname(os.path.dirname(query_name)),'masks', query_id) + '.png'
#         support_id = support_name.split('/')[-1].split('.')[0]
#         support_name = os.path.join(os.path.dirname(os.path.dirname(support_name)),'masks', support_id) + '.png'
#         # support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
#         # support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

#         query_mask = self.read_mask(query_name)
#         support_mask = self.read_mask(support_name)
#         # support_masks = [self.read_mask(name) for name in support_names]
        
#         return query_img, query_mask, support_img, support_mask

#     def read_mask(self, img_name):
#         mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
#         mask[mask < 128] = 0
#         mask[mask >= 128] = 1
#         return mask
    
#     def sample_episode(self, idx):
#         query_name = self.img_metadata[idx]
#         class_name = query_name.split('/')[-3]
#         support_names = []
#         support_tmp=[]
#         for file in os.listdir(os.path.dirname(query_name)):
#             if file.endswith(('.jpg')):
#                 support_tmp.append(os.path.join(os.path.dirname(query_name), file))

#         while True:  
#             support_name = random.choice(support_tmp)
            
#             if query_name != support_name: support_names.append(support_name)
#             if len(support_names) == self.kshot: break
            
#         return query_name, support_names[0], class_name 
    
#     def build_img_metadata(self):
#         img_metadata = []

#         for class_name in self.classes:
#             if self.trn:
#                 class_path = os.path.join(self.root_dir, 'trn', class_name)
#             else:
#                 class_path = os.path.join(self.root_dir, 'val', class_name)

#             folder_path = os.path.join(class_path, 'imgs')
            
#             img_paths = sorted(os.listdir(folder_path))
#             for img_path in img_paths:
#                 if os.path.basename(img_path).split('.')[1] == 'jpg':
#                     img_metadata.append(os.path.join(folder_path,img_path))
                    
#         return img_metadata
#数据集加载实验
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch

# class Compose(A.Compose):
#     def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
#         super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)
    
#     def __call__(self, image, mask):
#         augmented = super().__call__(image=np.array(image), mask=np.array(mask))
#         return augmented['image'], augmented['mask']
class Compose(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)
    
    def __call__(self, image):
        augmented = super().__call__(image=np.array(image))
        return augmented['image']
    
class MAMLDataset(Dataset):
    def __init__(self, root_dir, img_size, trn, kshot=1, k_qry=1, val_class=None, mode='trn'):
        self.root_dir = root_dir
        self.trn = trn
        self.q = k_qry
        self.mode = mode
        if self.trn:
            self.classes = sorted([folder for folder in os.listdir(os.path.join(root_dir, 'trn')) if not folder.startswith('.')])
        else:
            self.classes = [val_class]

        self.img_metadata = self.build_img_metadata()
        self.dirs = [os.path.join(os.path.join(root_dir,'trn'), class_name) for class_name in self.classes]
        self.kshot=kshot
        self.transform = Compose([
            A.Resize(img_size, img_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.pytorch.transforms.ToTensorV2(),
        ])
        
    def __len__(self):
        if self.trn:
            return len(self.img_metadata)// 3 // self.q
        else:
            return len(self.img_metadata)

    def __getitem__(self, idx):
        if self.trn:
            q_images = []
            q_masks = []
            s_images = []
            s_masks = []
            q_names = []
            s_names = []
            for dir in self.dirs:
               
                image_dir= os.path.join(dir, 'imgs')
                q_mask_dir = os.path.join(dir, 'masks')
                files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
                q_selected_files = files[idx*self.q:idx*self.q+self.q]
                #print("dir:{}, idx*self.q:{}, idx*self.q+self.q:{}".format(dir, idx*self.q, idx*self.q+self.q))
                q_images_class = []
                q_mask_class = []
                for file in q_selected_files:
                    q_image = Image.open(os.path.join(image_dir, file)).convert('RGB')
                    
                    q_names.append(os.path.join(image_dir, file))
                    
                    #print("idx:{}, query dir: {}, fileName: {} ".format(idx, dir, os.path.join(image_dir, file)))
                    
                    q_image = self.transform(q_image)
                    q_images_class.append(q_image)
                    q_mask = self.read_mask(os.path.join(q_mask_dir, os.path.splitext(file)[0]+".png"))
                    q_mask = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0).float(), q_image.size()[-2:], mode='nearest').squeeze()
                    q_mask_class.append(q_mask)
                q_images.append(torch.stack(q_images_class))
                q_masks.append(torch.stack(q_mask_class))
                s_selected_files=[]
                s_images_class=[]
                s_mask_class=[]
                while True:  
                    support_file = random.choice(files)
                    if support_file not in q_selected_files: s_selected_files.append(support_file)
                    if len(s_selected_files) == self.kshot: break

                for file in s_selected_files:
                    s_image = Image.open(os.path.join(image_dir, file)).convert('RGB')
                    
                    #print("support dataset: {}, fileName: {} ".format(dir, os.path.join(image_dir, file)))
                    
                    s_image = self.transform(s_image)
                    s_images_class.append(s_image)
                    s_mask = self.read_mask(os.path.join(q_mask_dir, os.path.splitext(file)[0]+".png"))
                    s_mask = F.interpolate(s_mask.unsqueeze(0).unsqueeze(0).float(), s_image.size()[-2:], mode='nearest').squeeze()
                    s_mask_class.append(s_mask)
                s_images.append(torch.stack(s_images_class))
                s_masks.append(torch.stack(s_mask_class))
            query_img = torch.stack(q_images)
            query_mask = torch.stack(q_masks)
            support_img = torch.stack(s_images)
            support_mask = torch.stack(s_masks)
        else: # VAL
            query_name, support_name, class_name = self.sample_episode(idx)
            query_img, query_mask, support_img, support_mask = self.load_frame(query_name, support_name)
            query_img = self.transform(query_img)
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

            support_img = self.transform(support_img)
            support_mask = F.interpolate(support_mask.unsqueeze(0).unsqueeze(0).float(), support_img.size()[-2:], mode='nearest').squeeze()
        
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_img': support_img,
                 'support_mask': support_mask,
                 }
        return batch
    def load_frame(self, query_name, support_name):
        query_img = Image.open(query_name).convert('RGB')
        support_img = Image.open(support_name).convert('RGB')
        # support_imgs = [Image.open(name).convert('RGB') for name in support_names]
        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(os.path.dirname(query_name)),'masks', query_id) + '.png'
        support_id = support_name.split('/')[-1].split('.')[0]
        support_name = os.path.join(os.path.dirname(os.path.dirname(support_name)),'masks', support_id) + '.png'
        # support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        # support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_mask = self.read_mask(support_name)
        # support_masks = [self.read_mask(name) for name in support_names]
        
        return query_img, query_mask, support_img, support_mask

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask
    
    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_name = query_name.split('/')[-3]
        support_names = []
        support_tmp=[]
        for file in os.listdir(os.path.dirname(query_name)):
            if file.endswith(('.jpg')):
                support_tmp.append(os.path.join(os.path.dirname(query_name), file))

        while True:  
            support_name = random.choice(support_tmp)
            
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.kshot: break
            
        return query_name, support_names[0], class_name 
    
    def build_img_metadata(self):
        img_metadata = []

        for class_name in self.classes:
            if self.trn:
                class_path = os.path.join(self.root_dir, 'trn', class_name)
            else:
                class_path = os.path.join(self.root_dir, 'val', class_name)
            if self.mode == 'test':
                class_path = os.path.join(self.root_dir, 'test', class_name)
            folder_path = os.path.join(class_path, 'imgs').replace('\r', '')
        
            
            img_paths = sorted(os.listdir(folder_path))
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(os.path.join(folder_path,img_path))
                    
        return img_metadata
        
        
        
        
        
        
        
        
        