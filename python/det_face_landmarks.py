import cv2
import numpy as np
from utils import priorbox, decode, decode_landm, align_img


class retinaface:
    def __init__(self,):
        self.detector = cv2.dnn.readNet("weights/retinaface_resnet50.onnx")
        self.det_inputsize = 512    ###输入正方形
        self.lmks_model = cv2.dnn.readNet("weights/landmark.onnx")
        self.lmks_inputsize = 224    ###输入正方形
        self.prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(self.det_inputsize, self.det_inputsize))
        self.variance = [0.1, 0.2]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        self.enlarge_ratio = 1.35
        self.lmk_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32).reshape((1, 1, 3))
        
        self.lm3d_std = np.array([[-0.31148657, 0.29036078, 0.13377953],
                                [0.30979887, 0.28972036, 0.13179526],
                                [0.0032535, -0.04617932, 0.55244243],
                                [-0.25216928, -0.38133916, 0.22405732],
                                [0.2484662, -0.38128236, 0.22235769]], dtype=np.float32)

    def preprocess(self, rgb_image):
        height, width = rgb_image.shape[:2]
        temp_image = rgb_image.copy()
        if height > width:
            scale = width / height
            new_width = int(self.det_inputsize * scale)
            new_height = self.det_inputsize
            temp_image = cv2.resize(temp_image, (new_width, new_height))
        else:
            scale = height / width
            new_height = int(self.det_inputsize * scale)
            new_width = self.det_inputsize
            temp_image = cv2.resize(temp_image, (new_width, new_height))

        input_img = (temp_image.astype(np.float32) / 255.0 - self.mean) / self.std
        x_min_pad = (self.det_inputsize - temp_image.shape[1]) // 2
        y_min_pad = (self.det_inputsize - temp_image.shape[0]) // 2
        input_img = cv2.copyMakeBorder(input_img, y_min_pad, self.det_inputsize - temp_image.shape[0] - y_min_pad, x_min_pad, self.det_inputsize - temp_image.shape[1] - x_min_pad, cv2.BORDER_CONSTANT, value=0)
        return input_img, x_min_pad, y_min_pad

    def predict(self, rgb_image, conf_threshold=0.7, nms_threshold=0.4):
        original_height, original_width = rgb_image.shape[:2]
        img, x_min_pad, y_min_pad = self.preprocess(rgb_image)
        input_tensor = cv2.dnn.blobFromImage(img)
        self.detector.setInput(input_tensor)
        # Perform inference on the image
        conf, land, loc = self.detector.forward(self.detector.getUnconnectedOutLayersNames())

        boxes = decode(loc[0], self.prior_box, self.variance)
        boxes *= self.det_inputsize
        scores = conf[0][:, 1]
        landmarks = decode_landm(land[0], self.prior_box, self.variance)
        landmarks *= self.det_inputsize

        # ignore low scores
        valid_index = scores > conf_threshold
        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        order = np.argsort(scores)[::-1]

        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold).flatten()
        if len(keep)==0:
            print('no face detect')
            return [], [], []
        
        boxes = boxes[keep].astype(np.int32)
        landmarks = landmarks[keep].reshape((-1, 2))
        scores = scores[keep]
        
        pad_border = np.array([[x_min_pad, y_min_pad]]) ###合理使用广播法则
        boxes[:, :2] -= pad_border
        landmarks -= pad_border
        boxes[:, 2:] += boxes[:, :2]  ###[x,y,w,h]转到[xmin,ymin,xmax,ymax]

        resize_coeff = max(original_height, original_width) / self.det_inputsize
        boxes = (boxes * resize_coeff).astype(np.int64)
        landmarks = (landmarks * resize_coeff).astype(np.int64)
        
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_width - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_height - 1)
        num_box = boxes.shape[0]
        landmarks = landmarks.reshape((num_box, -1, 2))

        return boxes, scores, landmarks

    def process_img(self, img_resize):
        img_resize = (img_resize.astype(np.float32) - self.lmk_mean) / 255.0
        input_tensor = cv2.dnn.blobFromImage(img_resize)
        self.lmks_model.setInput(input_tensor)
        # Perform inference on the image
        output = self.lmks_model.forward(self.lmks_model.getUnconnectedOutLayersNames())[0]
        output *= self.lmks_inputsize
        return output

    def infer(self, rgb_image):
        # rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, _, _ = self.predict(rgb_image)
        height, width, _ = rgb_image.shape
        landmarks = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i, :]
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) * 0.5
            cy = (y2 + y1) * 0.5
            sz = max(h, w) * self.enlarge_ratio

            x1 = cx - sz * 0.5
            y1 = cy - sz * 0.5
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2), :]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(crop_img, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (self.lmks_inputsize, self.lmks_inputsize))

            base_lmks = self.process_img(crop_img)
            inv_scale = sz / self.lmks_inputsize

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][0] = base_lmks[0, idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][1] = base_lmks[0, idx * 2 + 1] * inv_scale + trans_y1

            x1 = np.min(affine_base_lmks[:, 0])
            y1 = np.min(affine_base_lmks[:, 1])
            x2 = np.max(affine_base_lmks[:, 0])
            y2 = np.max(affine_base_lmks[:, 1])

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) * 0.5
            cy = (y2 + y1) * 0.5

            sz = max(h, w) * self.enlarge_ratio

            x1 = cx - sz * 0.5
            y1 = cy - sz * 0.5
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(crop_img, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT,
                                              value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (self.lmks_inputsize, self.lmks_inputsize))

            base_lmks = self.process_img(crop_img)
            inv_scale = sz / self.lmks_inputsize

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][0] = base_lmks[0, idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][1] = base_lmks[0, idx * 2 + 1] * inv_scale + trans_y1

            landmarks.append(affine_base_lmks)
        
        return boxes, landmarks

    def detect(self, srcimg):
        H = srcimg.shape[0]
        rgb_image = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        _, results_all = self.infer(rgb_image)
        if len(results_all)>0:
            results = results_all[0] # only use the first one
            landmarks=[]
            for idx in [74, 83, 54, 84, 90]:
                landmarks.append([results[idx][0], results[idx][1]])
            landmarks = np.array(landmarks).astype(np.float32)
            landmarks[:, -1] = H - 1 - landmarks[:, -1]

            trans_params, im, lm, _ = align_img(rgb_image, landmarks, self.lm3d_std)
            return trans_params, im
        else:
            print('no face detected! run original image')
            if np.array(srcimg).shape==(224,224,3):
                return None, rgb_image
            else:
                print('exit. no face detected! run original image. the original face image should be well cropped and resized to (224,224,3).')
                exit()

    def draw_detections(self, image, boxes,  landmarks):
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i, :]
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            for j in range(landmarks[i].shape[0]):
                cv2.circle(image, (int(landmarks[i][j, 0]), int(landmarks[i][j, 1])), 2, (0, 255, 0), thickness=-1)
        return image


if __name__ == '__main__':
    mynet = retinaface()
    imgpath = "testimgs/1.jpg"
    srcimg = cv2.imread(imgpath)
    rgb_image = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)

    boxes, landmarks = mynet.infer(rgb_image)
    dstimg = mynet.draw_detections(srcimg.copy(), boxes, landmarks)
    trans_params, im = mynet.detect(srcimg)
    print(im.shape)

    winName = 'retinaface in opencv-dnn'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.namedWindow("im", 0)
    cv2.imshow("im", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()