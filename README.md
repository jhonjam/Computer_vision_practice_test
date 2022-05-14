# Segmentação e Detecção de objetos com CNN
Implementação de segmentação e detecção de objetos com CNN

## Conteúdo
Este repositório contém uma implementação de segmentação semântica (Unet) para parafusos e um modelo de detecção de objetos (YOLOv4) para cogumelos. Os pesos estão disponíveis neste [link](https://drive.google.com/drive/folders/1ChtI9I-5SVqF6m0g9xwo6jp6fp71C7Yh?usp=sharing).



### Requerimentos e bibliotecas 
Requerimentos de instalação 

```bash
pip install -r requirements.txt
```

# Modelo de Detecção de objetos

## Métricas do Modelo de Detecção (YOLOv4 - Darknet)
Foram usadas as mtricas de Coco para avaliar o modelo de detecção.

| Modelo        | TP            | FP             | mAP (%) | 
| ------------- |:-------------:| :-------------:|-----:   |
| YOLOv4        | 60            | 54             |    70   |




## Inferência 
Para realizar a inferência do modelo de detecção pré-treinado no conjunto de teste , execute:

```bash
!python main_detection.py \
 --weight_folder "your_path/files_yolov4/" \
 --path_rgb_image "your_path/img_rgb_detection/" \
 --res_dir "your_path/results_object_detection"
```

