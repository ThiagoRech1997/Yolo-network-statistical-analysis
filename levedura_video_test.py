import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import darknet
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import csv
os.environ['GTK_VERSION'] = '3'

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="video_cal_mar15:5:14.avi",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="./backup/yolov4_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov4-levedura.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
    else:
        image = image_or_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)

    
def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path



def main():
    args = parser()
    check_arguments_errors(args)

    random.seed(4)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
#-----------------------------

    total_lev_detected = 0
    prev = time.time() 

    bins = np.linspace(0, 10, 100)

    listaDadosHist = []

    #escala microscopio
    #https://openwetware.org/wiki/Methods_to_determine_the_size_of_an_object_in_microns
    
    #fator de escala 40x (Belini)
    fEscala = 0.1632

    # abrindo a camera
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    now = datetime.datetime.now()

    j=0

    total_frames = 0

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    frames = 0
    frame_detect = 0
    temperatura = 30
    flag_criar_csv = 0
    
    #dados referente ao intervalo 20 min
    qtd_frames_distinct_contagem = 1
    media_intervalo = 0
    contagem_intervalo =0

    while(True):

        now = datetime.datetime.now()
        with open('csv/experimento_' + str(now.month) + "_" + str(now.year) + str(now.hour)+":"+str(now.minute)+":"+str(now.second)+ '.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Tamanho","Hora","Frame_Corrido","Frame_Deteccao","Probabilidade","Temperatura","Total_Levedura","Media_frame_deteccoes"])
            
            while (1):
                frames += 1
                ##Camera
                ret, frame = cap.read() 
                img = frame
                if ret:
                    # Access the image data
                    #cv2.imshow('Captura', cv2.resize(img,(width,height)))
                    

                    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 0] # sem compressao 
                    cv2.imwrite('image.png', img,compression_params)
                    image_name = 'image.png'
                    prev_time = time.time()
                    image, detections = image_detection(image_name, network, class_names, class_colors, args.thresh)
                    
                    #-----Antigo if detection--------------------------------
                    qtd_frames_distinct_contagem +=1

                    frame_detect +=1
                    now = datetime.datetime.now()
                    
                    for label, confidence, bbox in detections:
                        x,y,w,h = bbox
                        #calculo do tamanho das leveduras. maior medida vezes a escala (fator pixel por microns dado pela camera e lente microscopio)
                        if(w>h):
                            listaDadosHist.append((int)(w*fEscala))
                            writer.writerow([str(round(w*fEscala,2)).replace(".",","),str(now.hour)+":"+str(now.minute)+":"+str(now.second), str(frames), str(frame_detect), str(confidence).replace(".",","), temperatura,"",""])
                        else:
                            listaDadosHist.append((int)(h*fEscala))
                            writer.writerow([str(round(h*fEscala,2)).replace(".",","),str(now.hour)+":"+str(now.minute)+":"+str(now.second), str(frames), str(frame_detect), str(confidence).replace(".",","), temperatura,"",""])
                            
                        print(bbox)
                        total_lev_detected += 1
                        #por intervalo excel
                        contagem_intervalo += 1
                        
                    plt.subplot(1, 2, 1)
                    plt.axis('off')
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.title('Inference')
                    plt.subplot(1, 2, 2)
                    plt.hist(listaDadosHist, bins, alpha=0.5)
                    plt.savefig("resultado_inferencia/img_"+str(j)+".png")

                    #-----------------------------------------------------------------

                    if args.save_labels:
                        save_annotations(image_name, image, detections, class_names)
                    darknet.print_detections(detections, args.ext_output)
                    fps = int(1/(time.time() - prev_time))
                    print("FPS: {}".format(fps))

                    if not args.dont_show:
                        cv2.imshow('Inference', image)
                                               
                        compress_params = [cv2.IMWRITE_PNG_COMPRESSION, 0] # sem compressao 
                        cv2.imwrite("inferencias/Inferencia_"+str(now.hour)+":"+str(now.minute)+":"+srt(now.second)+".png",image,compress_params)
                        #cv2.waitKey(1)  

                    j+=1
                    #img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# converteu pra grayscale
                    k = cv2.waitKey(1)
                    if k == 27:
                        break
                    #time.sleep(0.1) # aguRDA 100 ms ate proximo frame
                    now = datetime.datetime.now()
                    #criando arquivo cuda_set_device
                    #flag serve para evitar a criacao de arquivos csv durante a duracao do minuto vigente (60 arquivos) para somente 1
                    if((now.minute == 0) and flag_criar_csv==0):
                        flag_criar_csv=1
                        writer.writerow(["","", "", "", "", "" ,str(contagem_intervalo),str(round(contagem_intervalo/qtd_frames_distinct_contagem,2)).replace(".",",")])
                        #zerando dados por intervalo
                        qtd_frames_distinct_contagem = 1
                        media_intervalo = 0
                        contagem_intervalo =0
                        break
                    elif((now.minute == 1) and flag_criar_csv==1):
                        flag_criar_csv=0
    
                    if((total_frames % 100) == 0):
                        now = datetime.datetime.now()
                        plt.hist(listaDadosHist , bins, alpha=0.5)
                        #plt.savefig("graph.png", transparent = True)
                        plt.savefig("graficos/graph_"+str(now.hour)+":"+str(now.minute)+":"+str(now.second)+".jpg")
                        plt.close()
                        #plt.show()
                        #plt.pause(0.1)
                        
                    print('total leveduras ',total_lev_detected)
                    
                    total_frames +=1
                
    cv2.destroyAllWindows()

#-----------------------------


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
