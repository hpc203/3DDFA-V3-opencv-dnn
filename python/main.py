import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import time
from det_face_landmarks import retinaface
from face_reconstruction import face_model
from io_ import visualize


if __name__=='__main__':
    args = {"iscrop": True, "detector": "retinaface", "ldm68": True, "ldm106": True, "ldm106_2d": True, "ldm134": True, "seg": True, "seg_visible": True, "useTex": True, "extractTex": True, "backbone": "resnet50"}
    imgpath = "testimgs/1.jpg"

    facebox_detector = retinaface()
    recon_model = face_model(args)

    srcimg = cv2.imread(imgpath)
    
    a = time.time()
    trans_params, im = facebox_detector.detect(srcimg)
    results = recon_model.forward(im)
    b = time.time()
    print(f'#### one image Total waste time: {(b-a):.3f}s')

    my_visualize = visualize(results, args)
    
    img_name = os.path.splitext(os.path.basename(imgpath))[0]
    save_path = os.path.join(os.getcwd(), 'results', img_name)
    os.makedirs(save_path, exist_ok=True)
    my_visualize.visualize_and_output(trans_params, srcimg, save_path, img_name)
    
