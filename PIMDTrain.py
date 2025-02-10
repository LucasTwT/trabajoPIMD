import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

#Elección del dispositivo de ejecución:

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Imagefolder
class ImageFolder(nn.Module): # clase que hereda del modulo nn.Module
    def __init__(self, root_dir, transforms = None): # constructor
        super(ImageFolder, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.data = [] # [(ruta_x, label_x), ....., (ruta_y, label_y)]
        self.class_names = os.listdir(root_dir)
        
        for index, name in enumerate(self.class_names): 
		        # se extraen las rutas de los archivos
            files = os.listdir(os.path.join(root_dir, name))
            # se realiza el label encoding
            self.data += list(zip(files, [index]*len(files)))
    
    def __len__(self): 
        """
        permite obtener la longitud del ImageFolder el 
        cual sera igual a la longitud del atributo data 
        """
            
        return len(self.data)
        
    def __getitem__(self, index):
        """
        permite obtener un dato como si fuera un array ImageFolder[index]
        """
        img_file, label = self.data[index] # ruta del archivo y etiqueta
        # ruta completa
        root_and_dir = os.path.join(self.root_dir, self.class_names[label]) 
        # carga de la imagen y conversion a un np.Array
        image = np.array(Image.open(os.path.join(root_and_dir, img_file))) 
        
        if self.transforms: # si transforms no es None
            agumentations = self.transforms(image=image) # se transforma la imagen
            image = agumentations['image'] # se actualiza la imagen
        return image, label # retorna la imagen y la etiqueta

def data_aug():
    transform = A.Compose([
	    A.Resize(224, 224), # redimension de la imagen a 244px * 244px
	    A.RandomCrop(width=150, height=200),  # ancho de 150px y altura de 200px
	    A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),
	    A.HorizontalFlip(p=0.6), # giro horizontal con probabilidad del 60%
	    A.VerticalFlip(p=0.3),# giro vertical con probabilidad del 30%
	    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25,p=0.3),
	    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
	    ToTensorV2(), #conversion a un tensor de pytorch
    ])
    return transform

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Capas convolucionales:
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        
        #Capas fully connect:
        self.fc1 = nn.Linear(64 * 25 * 18, 900)
        self.fc2 = nn.Linear(900, 60)
        self.fc3 = nn.Linear(60, 1)
        
        #self.dropout:
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2= nn.Dropout(0.25)
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        # x = bath_size, 3 (R, G, B), width=150, height=200
        x = F.relu(self.conv1(x)) # 16, 32, 32
        x = self.dropout1(x) 
        x = F.max_pool2d(x, 2, 2) # 16, 16, 16
        x = F.tanh(self.conv2(x)) # 32, 16, 16
        x = self.dropout2(x) 
        x = F.max_pool2d(x, 2, 2) # 32, 8, 8
        x = F.tanh(self.conv3(x)) # 64, 8, 8
        x = F.max_pool2d(x, 2, 2) # 64, 4, 4
        x = x.view(-1, 64 * 25 * 18) # redimensionamos el tensor -> batch_size, 1024
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.sigmoid(self.fc3(x))
        return x

def main() -> None:
    print(device)
    transform = data_aug()
    #Image folders
    train_img_folder = ImageFolder(
						r'Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Data\ExpresionesDeEstres\KDEF\KDEF\Train',
                            transforms=transform)
    test_img_folder = ImageFolder(
                            r'Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Data\ExpresionesDeEstres\KDEF\KDEF\Test',
                            transforms=transform)
    #dataloaders:
    train_dataloader = DataLoader(train_img_folder, batch_size=32, num_workers=10, shuffle=True)
    test_dataloader = DataLoader(test_img_folder, batch_size=32, num_workers=10, shuffle=False)
    
    model = CNN().to(device)
    lr = .001
    epochs = 3
    loss_funct = nn.BCELoss()
    optimizer = opt.SGD(model.parameters(), lr)
    error = []
    
    for epoch in tqdm(range(epochs)):
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device).float().view(-1, 1)
            model.train()
            pred = model(x)
            loss = loss_funct(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f' Época número -> {epoch} con error de: {loss.item()*100:.2f}')
        error.append(loss.item())

    #Gráfica y evaluación en fasse de training 
    
    with open(r'Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Pesos\LossTrainPIMD.json', 'w') as f:
        json.dump(error, f)
    
    plt.plot([index for index in range(len(error))], error, color='r', label=f'Error: {error[-1]*100:.2f}%')
    plt.legend(title='Leyenda: ')
    plt.title("Error en fase de training: ")
    plt.show()
    
    #Guardado de pesos:
    opc = input("Red neruonal entrenada introduce un True si quieres guardar los pesos: ")
    
    if opc:
        ruta_guardado = r'Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Pesos\PIMD.pth'
        torch.save(model.state_dict(), ruta_guardado)
    
if __name__ == '__main__':
    main()