from torch.utils.data import DataLoader
import torch
from torchvision import datasets,transforms
import os

epochs = 50
batch_size = 64
torch.manual_seed(17)

dummy_batch = DataLoader(
    datasets.ImageFolder('D:\dataset\lfw', transform=transforms.ToTensor()),
    batch_size=1,
    shuffle=True
)
transforms_set = transforms.Compose([
    transforms.Resize(size=150),
    transforms.ToTensor(),
])
transforms_all = transforms.Compose([
    transforms.Resize(size=150),
    transforms.ToTensor(),
])

folders = os.listdir('D:\dataset\lfw')

# ignore hidden files
folders = [folder for folder in folders if folder[0]!='.']

if not os.path.exists('D:\dataset\\train'):
    os.mkdir('D:\dataset\\train')
if not os.path.exists('D:\dataset\\val'):
    os.mkdir('D:\dataset\\val')
'''
for folder in tqdm(folders):
    if not os.path.exists(f'D:\\dataset\\train\\{folder}'):
        os.mkdir(f'D:\\dataset\\train\\{folder}')
    if not os.path.exists(f'D:\\dataset\\val\\{folder}'):
        os.mkdir(f'D:\\dataset\\val\\{folder}')
    
    images = os.listdir(f'D:\dataset\lfw\{folder}')
    images = [image for image in images if folder[0]!='.']
    
    random.shuffle(images)
    for image in images[-6:]:
        shutil.copy(f'D:\dataset\lfw\{folder}\{image}', f'D:\dataset\\train\{folder}\{image}')
    for image in images[:-6]:
        shutil.copy(f'D:\dataset\lfw\{folder}\{image}', f'D:\dataset\\val\{folder}\{image}')

'''


train_loader = DataLoader(
    datasets.ImageFolder('D:\dataset\\train\\', transform=transforms_set),
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    datasets.ImageFolder('D:\dataset\\train\\', transform=transforms_set),
    batch_size=batch_size,
    shuffle=True
)

transforms_set = transforms.Compose([
    transforms.Resize(size=150),
    transforms.ToTensor(),
])