import argparse
import numpy as np
import cv2
import os
import pprint
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()


parser.add_argument(
    "--weight_folder",
    type=str,
    default="C:/Users/jhonj/Downloads/prova_vision_computacional/weight/",
    help="Path to the main folder containing the pre-trained weights",
)

parser.add_argument(
    "--cfg_folder",
    type=str,
    default="C:/Users/jhonj/Downloads/prova_vision_computacional/weight/",
    help="Path to the main folder containing the cfg file",
)


parser.add_argument(
    "--path_rgb_image",
    default='C:/Users/jhonj/Downloads/prova_vision_computacional/img_rgb_detection/',
    type=str,
    help="Path to the folder of image rgb for test",
)

parser.add_argument(
    "--res_dir",
    default="C:/Users/jhonj/Downloads/prova_vision_computacional/result",
    help="Path to the folder where the results is save",
)

parser.add_argument(
    "--names",
    default="C:/Users/jhonj/Downloads/prova_vision_computacional/weight/",
    help="Path to the folder where is file names",
)


def color_classe(classes):
    '''
    Function to define different color for each objects
    Args:
    ---------
        classes: class identified by the model
    Return:
    ---------
        color: representative color
    '''
    if classes == 0: # Cantarelo 
        color = (255,100,100)
        return color
    elif classes == 1:  # Chicken
        color = (100,100,255)
        return color


# Get the names of all the layers in the network
def obtener_nombre_salida(net): 
    name_layer = net.getLayerNames()
    #ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return [name_layer[i - 1] for i in net.getUnconnectedOutLayers()]


def etiquetado(frame,out):
    '''
    function to extract information from the coordinates of objects
    Args:
    ---------
        frame: current image
        out : information provided by the detection model 
    Return:
    ---------
        frame_out: output image with the respective detections
    '''

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIDs = []
    confidences = []
    rectangulos = []
    
    for out in out:
        for detection in out:
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence> 0.1: 
                cx = int(detection[0]*frameWidth)
                cy = int(detection[1]*frameHeight)
                w = int(detection[2]* frameWidth)
                h = int(detection[3]*frameHeight )
                x = int(cx - w/2)
                y = int(cy - h/2)
                
                classIDs.append(classID)# extraer el nombre de la clase que se detecto
                confidences.append(float(confidence))# guaradr que tan confiavle fue la deteccion
                rectangulos.append([x, y, w, h])

    # NMS technique to eliminate redundant detections 
    indices = cv2.dnn.NMSBoxes (rectangulos,confidences, 0.25, 0.40 )
   
    # creating bounding boxes in the image
    for i in range(len(rectangulos)):
        if i in indices:
            #i = i[0]
            x,y,w,h= rectangulos[i]
            area= ((x+w)-x)*((y+h)-y)
            if area>400:
                x1 = int(w / 2)
                y1 = int(h / 2)
                cx = x + x1
                cy = y + y1    
                label11 = '%.1f' % confidences[i]

                if classes:
                    assert(classIDs[i] < len(classes))
                    label1 = '%s%s' % (classes[classIDs[i]], label11)

                bbox_mess = '%s%s' % (classes[classIDs[i]], label11)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color_classe(classIDs[i]),2)
                fontScale = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, bbox_mess, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                # etiqueta
                #https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/utils.py
                #bbox_thick = int(3 * (400 + 600) / 600)
               # bbox_mess = '%s%s' % (classes[classIDs[i]], label11)
                #t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=bbox_thick // 2)[0]  # bbox_thick // 2
                #c3 = (x + t_size[0], y - t_size[1] - 3)
                #cv2.rectangle(frame, (x,y), (x+w, y+w), (255, 0, 0), -1)
                #cv2.rectangle(frame, (x,y), (np.float32(c3[0]), np.float32(c3[1])), color_classe(classIDs[i]), -1) #filled
                #cv2.putText(frame, bbox_mess, (x, np.float32(y - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0),1 , lineType=cv2.LINE_AA)
    return frame




# Funcion general
def main(config):
    global classes

    # network files (CNN)
    weight = os.path.join(config.weight_folder, "yolov4.weights")
    cfg = os.path.join(config.cfg_folder, "yolov4.cfg")
    coconames = os.path.join(config.names, "coco.names")
    
    net = cv2.dnn.readNet(weight, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    classes = []
    with open(coconames,'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Inference on test images
    list_img_rgb = os.listdir(config.path_rgb_image) 
    for i,j in enumerate(list_img_rgb):
        img = cv2.imread(config.path_rgb_image + j)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Network parameters 
        blob = cv2.dnn.blobFromImage(img, 1/255, (512, 512), [0,0,0], 1, crop = False)
        net.setInput(blob)
        salida = net.forward(obtener_nombre_salida(net))

        # Detections by cnn
        detections = etiquetado(img,salida,)
        print(config.res_dir)
        #cv2.imwrite(config.res_dir + 'detecion_img_{}.png'.format(i), detections)
        #cv2.imwrite(os.path.join(config.res_dir , 'detecion_img_{}.png'.format(i)),detections)
        cv2.imwrite('detecion_img_{}.png'.format(i),detections)





if __name__ == "__main__":
    config = parser.parse_args()

    pprint.pprint(config)
    main(config)