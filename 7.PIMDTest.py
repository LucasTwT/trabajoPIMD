import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from PIMDTrain import CNN, ImageFolder, data_aug
from tqdm import tqdm
import json
import os
import numpy as np

# Elección del dispositivo:
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def main() -> None:
    ruta_guardado = r'Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Pesos\PIMD.pth' # ruta donde estan los pesos del modelo
    transform = data_aug()
    model = CNN().to(device) # cargamos la arquitectura del modelo
    model.load_state_dict(torch.load(ruta_guardado)) # se cargan los pesos
    loss_funct = nn.BCELoss() # funcion de pérdida
    #Datos de testing
    ruta_data = r'Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Data\ExpresionesDeEstres\KDEF\KDEF\Test'
    test_img_folder = ImageFolder(
                            ruta_data,
                            transforms=transform)
    
    test_dataloader = DataLoader(test_img_folder, batch_size=32, num_workers=10, shuffle=False)
    
    # entrenamiento en fase de testing
    model.eval()
    preds = []
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x, y = x.to(device).float(), y.to(device).float().view(-1, 1)
            pred = model(x)
            loss = loss_funct(pred, y)
            for p in pred:
                preds.append(p.to('cpu'))
    
    loss = loss.to('cpu').numpy()
    print(f'Error en fase de testing de: {loss*100:.2f}%')
    
    # validación con gráficos:
        
    with open(r'Scripts\Rama1\EjericiosPrácticos\5.TodoJunto\CNN\Pesos\LossTrainPIMD.json', 'r') as f:
        error_train = json.load(f)
        
    plt.plot([index for index in range(len(error_train))], error_train, color='r', label=f'Error en train: {error_train[-1]*100:.2f}%')
    plt.plot([index for index in range(len(error_train))], [loss for _ in range(len(error_train))], color='b', label=f'Error en test: {loss*100:.2f}%')
    plt.legend(title='Leyenda:')
    plt.title('Error en train vs Error en test:')
    
    # Selección aleatoria de 10 imágenes del set de testing comparación entre la predicción versus el target
    num_imgs = 5
    sub_directorios = os.listdir(ruta_data)
    fig, ax = plt.subplots(len(sub_directorios), num_imgs, figsize=(18,8))
    print(len(preds))
    preds = np.round(preds)
    preds = ["No estresado" if pred == 0 else "Estresado" for pred in preds]
    for dir in range(len(sub_directorios)):
        path_sub = os.path.join(ruta_data, sub_directorios[dir])
        dir_img = os.listdir(path_sub)
        for img in range(num_imgs):
            num_aleatorio = np.random.randint(0, len(dir_img)-1)
            imagen = np.asarray(Image.open(os.path.join(path_sub, dir_img[num_aleatorio])))
            ax[dir, img].imshow(imagen)
            ax[dir, img].legend(title=f'Target -> {sub_directorios[dir]}\nPredicción -> {preds[num_aleatorio]}')
            ax[dir, img].set_title(sub_directorios[dir])
    plt.show()
    
if __name__ == '__main__':
    main()