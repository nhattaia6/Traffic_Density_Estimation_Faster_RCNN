import cv2
from threading import Thread
import threading
import concurrent.futures
from time import sleep
import os
from datetime import datetime
from copy import deepcopy
import base64

# lib for sms sending
from twilio.rest import Client

# Lib for email sending
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)


# Load pretrained model - prepare for prediction
from frcnn import *
# from resnet_frcnn import *

from draw import *
from database import *

###############
import numpy as np

def rotateImage(image, angle):
    """
    Rotate image with X angle
    """
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image
################

def change_label_name(label):
    if label=='2-wheel':
        return 'bike'
    elif label=='4-wheel':
        return 'car'
    else:
        return label

# Create subclass of Thread, get returned of thread
class ThreadWithReturnValue(Thread):
    """
    Override Thread class, get returned of thread
    """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return

class Predict_video():
    """
    Predict video with trained model
    """
    def __init__(self):
        """Load pretrained model and config"""
        ########### load config of model ##############
        base_path = 'E:\LVTN\Sample\Faster_RCNN_for_Open_Images_Dataset_Keras-master/'
        config_file_path = 'model_vgg_config_scenario1.pickle'
        # config_file_path = 'scenario_bak/new_256_model_vgg_config.pickle'
        # config_file_path = 'scenario_bak/resnet/new_256_model_vgg_config_resnet.pickle'
        # config_file_path = 'scenario_bak/resnet/_256_300/new_256_model_vgg_config_resnet.pickle'
        config_output_filename = os.path.join(base_path, config_file_path)
        with open(config_output_filename, 'rb') as f_in:
            self.C = pickle.load(f_in)

        # turn off any data augmentation at test time
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False
        self.C.model_path = base_path+'model/model_frcnn_vgg_scenario1_19.hdf5'
        # self.C.model_path = base_path+'scenario_bak/new_256_model_frcnn_vgg_30.hdf5'
        # self.C.model_path = base_path+'scenario_bak/resnet/new_256_model_frcnn_resnet.hdf5'
        # self.C.model_path = base_path+'scenario_bak/resnet/_256_300/new_256_model_frcnn_resnet.hdf5'
        print("aaaaaa", self.C.model_path)

        ############### load model ###############
        num_features = 512 #1024 #

        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (VGG here, can be Resnet50, Inception, etc)
        shared_layers = nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        rpn_layers = rpn_layer(shared_layers, num_anchors)

        classifier = classifier_layer(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.C.class_mapping))

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(self.C.model_path))
        self.model_rpn.load_weights(self.C.model_path, by_name=True)
        model_classifier.load_weights(self.C.model_path, by_name=True)

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        # Switch key value for class mapping
        self.class_mapping = self.C.class_mapping
        self.class_mapping = {v: k for k, v in self.class_mapping.items()}
        for k, v in self.class_mapping.items():
            self.class_mapping[k] = change_label_name(v)
        print(self.class_mapping)
        class_to_color = {self.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.class_mapping}

    def set_up_scenario(self, process_every_x_frame=50, create_new_thread_every_x_frame=30, check_thread_result_every_x_frame=100, send_sms_every_x_frame=250, create_thread_for_x_frame_first=300, rotate_image=0, send_sms=0, send_email=0, min_status=5, max_status=10, ratio_car_per_bike=2, ratio_priority_per_bike=3):
        """
        Set up scenario for system
        Set up config for video (rotate, send SMS flag)
        Set up min and max for status (Low < min < Medium < max < Traffic jam)
        """
        self.process_every_x_frame              = process_every_x_frame
        self.create_new_thread_every_x_frame    = create_new_thread_every_x_frame
        self.check_thread_result_every_x_frame  = check_thread_result_every_x_frame
        self.send_sms_every_x_frame          = send_sms_every_x_frame
        self.create_thread_for_x_frame_first    = create_thread_for_x_frame_first
        self.rotate_image                       = rotate_image
        self.send_sms                           = send_sms
        self.send_email                         = send_email
        self.min_status                         = min_status
        self.max_status                         = max_status
        self.ratio_car_per_bike                 = ratio_car_per_bike
        self.ratio_priority_per_bike            = ratio_priority_per_bike

    def init_video(self, video_path, phone_number='0338684430', email='nhattaia6@gmail.com', max_distance=170, bbox_threshold=0.7):
        """Init video information"""
        self.video_path             = video_path
        self.max_distance           = max_distance
        self.bbox_threshold         = bbox_threshold
        self.phone_number           = phone_number
        self.email                  = email
    
        ######## setup video ###########
        self.cap_video = cv2.VideoCapture(self.video_path)
        self.cap_video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.total_frame = int(self.cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video path:", self.cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # define threads
        self.threads = []

        # First, create thread for 100 frames!
        self.create_thread(0, self.create_thread_for_x_frame_first)


        self.processed_images = {}

        # Init information of sms client
        self.account_sid = os.environ['TWILIO_ACCOUNT_SID']
        self.auth_token = os.environ['TWILIO_AUTH_TOKEN']
        self.client = Client(self.account_sid, self.auth_token) 

        # Client backup
        # self.client = Client('ACa763a7eaf3e5e50f625c8b9c8cf69ef2', 'bcca22ffc4cde1dc8b9c554ecc71ed6e')

        self.vid2 = cv2.VideoCapture(self.video_path)
        self.vid2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.thread_frame = self.create_thread_for_x_frame_first 

        self.conclusion_label = []
    
    def get_counter_area(self):
        """
        Draw area to count vehicle inside that area!
        Return list tuple are points on the area and a tuple is a point inside the area!
        """
        # Get image from video
        _, frame = self.cap_video.read()
        if frame is None:
            return False
        
        frame = cv2.resize(frame, (1280, 720))
        if self.rotate_image!=0:
            frame = rotateImage(frame, self.rotate_image)
        # # Scale 75% display
        # frame = self.rescale_frame(frame)
        gca = Get_counter_area(frame) 
        gca.set_up_mouse_call_back_event()
        self.points, self.center = gca.run()
        if self.points==False:
            print("Bye!")
            return False

        # Get threads result
        self.run_thread(0.3)
        self.get_result_t = ThreadWithReturnValue(target=self.get_thread_result)
        self.get_result_t.start()
        print("Waiting for first image processing...!")
        sleep(3)
        print("points:{}, center:{}".format(self.points, self.center))

    # phuong trinh duong thang: ax + by + c = 0
    def check_location_point_with_d(self, A,B,G):
        """
        Return G and Center point location with d(AB)
        if > 0: G and Center on the same side compare to AB
        if < 0: G and Center on the other side compare to AB
        if = 0: G or Center located on AB
        """   
        xA = A[0]
        yA = A[1]
        xB = B[0]
        yB = B[1]
        xG = G[0]
        yG = G[1]
        vtpt = [-(yB-yA), (xB-xA)]
        d=lambda x,y: (vtpt[0]* (x-xA) + vtpt[1]* (y-yA))
        dG = d(xG,yG)
        dE = d(self.center[0],self.center[1])
        return dG*dE

    def is_object_located_on_counter_area(self, obj_location):
        """
        If point is inside or located on the area -> return True
        else -> return False
        """
        for i in range(len(self.points)-1):
            if self.check_location_point_with_d(self.points[i], self.points[i+1], obj_location) <0:
                return False
        if self.check_location_point_with_d(self.points[0], self.points[-1], obj_location) <0:
            return False
        return True

    def rescale_frame(self, frame, percent=75):
        """
        Scale frame with X percent (default 75)
        return frame after scaled
        """
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
        return frame

    def format_img_size(self, img):
        """ formats the image size based on config
        Return image after resized and ratio
        """
        img_min_side = float(self.C.im_size)
        (height,width,_) = img.shape
            
        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio	

    def format_img_channels(self, img):
        """ formats the image channels based on config 
        Return image after formated
        """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self, img):
        """ formats an image for model prediction based on config 
        Return image after formated and ratio
        """
        img, ratio = self.format_img_size(img)
        img = self.format_img_channels(img)
        return img, ratio

    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(self, ratio, x1, y1, x2, y2):
        """ Transform the coordinates of the bounding box to its original size """
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)

    # Check old or new object
    def is_old(self, center_Xd, center_Yd, boxes):
        """ is old object? """
        for box_tracker in boxes:
            (xt, yt, wt, ht) = [int(c) for c in box_tracker]
            center_Xt, center_Yt = int((xt + (xt + wt)) / 2.0), int((yt + (yt + ht)) / 2.0)
            distance = math.sqrt((center_Xt - center_Xd) ** 2 + (center_Yt - center_Yd) ** 2)

            if distance < self.max_distance:
                return True
        return False


    def get_box_info(self, box):
        """ Get x,y,w,h,center_X,center_Y from x,y,w,h """
        (x, y, w, h) = [int(v) for v in box]
        center_X = int((x + (x + w)) / 2.0)
        center_Y = int((y + (y + h)) / 2.0)
        return x, y, w, h, center_X, center_Y


    def predict_one_image(self, img):
        """
        Predict one image with pretrained model
        Input: img
        Output: - list of bounding boxes (x1, y1, w, h)
                - list of bounding boxes (x1, y1, x2, y2)
                - list of keys
        """
        time_per_one = time.time()

        X, ratio = self.format_img(img)
        
        # switch column (0,1,2,3) -> (0,2,3,1)
        X = np.transpose(X, (0, 2, 3, 1))

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = self.model_rpn.predict(X)

        # Get bboxes by applying NMS 
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, self.C, change_K_format(K.image_data_format()), overlap_thresh=0.65)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//self.C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < self.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        # all_dets = []
        all_xywh = []
        # all_boxes = []
        all_keys = []

        for key in bboxes:
            bbox = np.array(bboxes[key])
            
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.25)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]

                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
                
                # all_boxes.append([real_x1, real_y1, real_x2, real_y2])

                # cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

                # textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                # all_dets.append((key,100*new_probs[jk]))

                cx = int((real_x1+real_x2)/2)
                cy = int((real_y1+real_y2)/2)

                # print("cx,cy", cx,cy)
                # print("kq:", self.is_object_located_on_counter_area((cx,cy)))
                
                # Show vehicle outside the area!
                if not(self.is_object_located_on_counter_area((cx,cy))):
                    continue

                w = real_x2 - real_x1
                h = real_y2 - real_y1
                all_xywh.append((real_x1, real_y1, w, h))

                all_keys.append([key,int(100*new_probs[jk])])

                # (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                # textOrg = (real_x1, real_y1-0)

                # cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
                # cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                # cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        print('Spend %0.2f seconds to process an image!' % ((time.time()-time_per_one)) )
        return all_xywh, all_keys #all_boxes

    def reprocess_image(self, img_idx):    
        """
        Process X frame first.
        Input cap of video, total frame of video and index of image want to process
        Output processed image and list of bounding boxes
        """
        cap_cache = cv2.VideoCapture(self.video_path)
        # Check if image index eq total frame of video
        if img_idx <= self.total_frame:
            vehicle_count = []
            # Jumping to the image index
            cap_cache.set(cv2.CAP_PROP_POS_FRAMES, img_idx)
            print('Position:', int(cap_cache.get(cv2.CAP_PROP_POS_FRAMES)))

            # Get image from video
            _, frame = cap_cache.read()
            if frame is None:
                return False
            
            # Resize image to 1280x720
            frame = cv2.resize(frame, (1280,720))

            # Rotae image
            if self.rotate_image:
                frame = rotateImage(frame, self.rotate_image)

            # Process image - Detect objects
            xywh_d, keys_d = self.predict_one_image(frame) #boxes_d, 
            boxes_d_enum = list(enumerate(xywh_d))

            for idx, box in boxes_d_enum:

                xd, yd, wd, hd, center_Xd, center_Yd = self.get_box_info(box)
                
                cv2.rectangle(frame, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 255), 2)

                label = keys_d[idx]
                textLabel = '{}'.format(label)
                cv2.putText(frame, textLabel, (xd, yd +20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

                #Ve hinh tron tai tam doi tuong
                cv2.circle(frame, (center_Xd, center_Yd), 4, (0, 255, 0), -1)

                # Count vehicle in counted area
                if (self.is_object_located_on_counter_area((center_Xd, center_Yd))):
                    cv2.rectangle(frame, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 0), 2)
                    vehicle_count.append(label[0])

            # Print title in the image
            title = 'frame {}'.format(img_idx)
            cv2.putText(frame, title, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)
            
            #cv2.imshow('video jumping', frame)
            
            # key = cv2.waitKey(10) & 0xFF
            # if key == 27:
            #     break

            return frame, vehicle_count
        else:
            return False

    # reprocess image
    def create_and_run_thread(self, from_frame, to_frame, sleep_time):
        """
        Create thread for image processing and start them
        Input: from frame, to frame, sleep time bettween each thread created
        Output: No output, but there threads created will begin
        """
        for i in range(from_frame, to_frame, self.process_every_x_frame): # change to total_frame if not test
            # creating thread
            
            t = ThreadWithReturnValue(target=self.reprocess_image, args=(i,))

            self.threads.append((t,i))

            # starting thread
            t.start() 
            print("thread {} started!".format(i))
            sleep(sleep_time)
            # break

    def create_thread(self, from_frame, to_frame):
        """
        Create thread for image processing but not start them
        Input: from frame, to frame, sleep time bettween each thread created
        Output: No output, but there threads created will begin
        """
        for i in range(from_frame, to_frame, self.process_every_x_frame): # change to total_frame if not test
            # creating thread
            
            t = ThreadWithReturnValue(target=self.reprocess_image, args=(i,))

            self.threads.append((t,i))

    def run_thread(self, sleep_time):
        for t,i in self.threads:
            # starting thread
            t.start() 
            print("thread {} started!".format(i))
            sleep(sleep_time)
            # break

    
    def get_thread_result(self):   
        """
        Get result of thread
        """ 
        for t, i in self.threads:
            print("len self.threads {}".format(len(self.threads)))
            # while t.join() is None:
            #     sleep(0.5)
            self.processed_images[i] = t.join()              
            print("got the result of thread {}!".format(i))
            if self.processed_images[i] is not None and self.threads.count((t,i))>0:
                print("remove thread (t,i)=({},{})".format(t,i))
                self.threads.remove((t,i))

    def get_conclusion_label(self, vehicle_count):
        """
        Get final result after X frame (example: 20 second)
        Input: list of vehicle counted every 20 second
        Output: Low, Medium or Traffic jam
        """
        bike = vehicle_count.count('bike')
        car = vehicle_count.count('car')
        priority = vehicle_count.count('priority')

        total_vehicle = bike + self.ratio_car_per_bike*car + self.ratio_priority_per_bike*priority

        if total_vehicle < self.min_status:
            status = 'Low'
        elif self.min_status <= total_vehicle < self.max_status:
            status = 'Medium'
        else:
            status = 'Traffic jam' 
        return status, bike, car, priority

    def run_predict_video(self):
        """
        Start video prediction
        """
        frame_count = 0
        while self.vid2.isOpened():
            # Get frame from video
            _, frame = self.vid2.read()
            if frame is None:
                break
            
            # Resize frame to 1280x720
            frame = cv2.resize(frame, (1280,720))

            # Rotae image
            if self.rotate_image:
                frame = rotateImage(frame, self.rotate_image)

            # Processed image every self.process_every_x_frame frames (will change 20 seconds)
            if frame_count % self.process_every_x_frame == 0:
                # Wait the result util got the result from thread
                while(self.processed_images.get(frame_count) is None):
                    if not self.get_result_t.is_alive():
                        self.get_result_t.run()
                        print("rerun get the result thread!")
                    print("Wait result of frame {}!".format(frame_count))
                    sleep(3)
                frame , vehicle_count = self.processed_images.get(frame_count)
                print("vehicle_count", vehicle_count)
                label_count = 'bike: {}      car: {}     priority: {}'.format(vehicle_count.count('bike'),vehicle_count.count('car'),vehicle_count.count('priority'))
                cv2.putText(frame, label_count, (100, 690), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

                # Add conclusion label
                status, bike , car, priority = self.get_conclusion_label(vehicle_count)
                self.conclusion_label.append(status)

                # Insert to database
                time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                time_temp = time_now.split(' ')
                date = time_temp[0]
                time = time_temp[1]
                print(date, time)

                base_dir = 'E:/LVTN/Sample/Faster_RCNN_for_Open_Images_Dataset_Keras-master/DB/Image/'
                img_name = '{}_{}.jpg'.format(date,time.replace(':','-'))
                img_path = base_dir + img_name
                
                db = Database()
                db.insert_data(date, time, bike, car, priority, status, img_path)

            else:
                title = 'frame {}'.format(frame_count)
                cv2.putText(frame, title, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)
                label_count = 'bike: {}      car: {}     priority: {}'.format(0,0,0)
                cv2.putText(frame, label_count, (100, 690), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)     

            # Create thread every self.process_every_x_frame frames to process image
            if (frame_count+1)%(self.create_new_thread_every_x_frame+1) == 0 and self.thread_frame <= 500:# and len(self.threads)<=5:
                self.create_and_run_thread(self.thread_frame, self.thread_frame + self.process_every_x_frame, 0)
                self.thread_frame +=self.process_every_x_frame
                # if not self.get_result_t.is_alive():
                #     self.get_result_t.run()
                #     print("rerun get the result thread!")

            # Check status of thread, rerun if they die!
            if (frame_count+1)%(self.check_thread_result_every_x_frame+1) == 0 and self.threads and not self.get_result_t.is_alive():
                self.get_result_t.run()
                print("rerun get the result thread!")

            # Send SMS and email every X frames
            if (frame_count+1)%(self.send_sms_every_x_frame+1) == 0 and (self.send_sms or self.send_email):
                # Get time
                time_now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                # Send sms
                if self.send_sms:
                    try:
                        sms_body = "HELLO, The current traffic condition is {} at {}!".format(max(self.conclusion_label, key=self.conclusion_label.count), time_now)
                        message = self.client.messages.create(body=sms_body ,from_='+12514283543', to='+84338684430')
                        
                        # SMS backup
                        # message = self.client.messages.create(body=sms_body ,from_='+13102376022', to='+84989128125')
                        print(message.sid, "\n", message.body)
                        print("Sent SMS!")
                    except Exception as e:
                        print("Error: ",e)
                        print("SMS sending failed!")
                        pass
                

                # Send email
                print("self.send_email", self.send_email)
                print("self.email", self.email)
                if self.send_email and self.email is not None:
                    print("ok")
                    # Get image file
                    last_file = db.get_last_image_path()
                    print("image file:",last_file)
                    if os.path.isfile(last_file):
                        with open(last_file, 'rb') as f:
                            data = f.read()
                            f.close()
                            encoded_file = base64.b64encode(data).decode()

                            attachedFile = Attachment(
                                FileContent(encoded_file),
                                FileName(img_name),
                                FileType('application/image'),
                                Disposition('attachment')
                            )
                    else:
                        attachedFile = False
                    to_email = self.email
                    print("list email", to_email)
                    sendgrid_api_key = os.environ.get('SENDGRID_API_KEY')
                    for email in to_email:
                        print("Send to email: ", email)
                        email_content = '''<H2>Hi <strong>{}</strong></H2>
                                            <p>The current traffic condition is <strong>{}</strong></p></br>
                                            <p>Time: <strong>{}</strong></p>
                                            </br>------------
                                            </br> -----------                                       
                                            <h4>Bui Nhat Tai - B1606838</h4>
                                            <h4>TRAFFIC DENSITY ESTIMATION SYSTEM</h4>
                                            </br>------------ 
                                            </br>------------ 
                                        '''.format(email, max(self.conclusion_label, key=self.conclusion_label.count), time_now)
                        message = Mail(
                            from_email='taib1606838@student.ctu.edu.vn',
                            to_emails=[email],
                            subject='NOTIFICATION FROM TRAFFIC DENSITY ESTIMATION SYSTEM',
                            html_content=email_content)
                        try:
                            if attachedFile:
                                message.attachment = attachedFile
                            # print("aaaaa", os.environ.get('SENDGRID_API_KEY'))
                            sg = SendGridAPIClient(sendgrid_api_key)
                            response = sg.send(message)
                            print(response.status_code)
                            print(response.body)
                            print(response.headers)
                            print("Sent email!")
                        except Exception as e:
                            print(e)
                            print("Email sending failed!")
                            pass
                    else:
                        email_content = '''<H2>Hi <strong>{}</strong></H2>
                                            <p>The current traffic condition is <strong>{}</strong></p></br>
                                            <p>Time: <strong>{}</strong></p>
                                            </br>------------
                                            </br> -----------                                       
                                            <h4>Bui Nhat Tai - B1606838</h4>
                                            <h4>TRAFFIC DENSITY ESTIMATION SYSTEM</h4>
                                            </br>------------ 
                                            </br>------------ 
                                        '''.format(to_email, max(self.conclusion_label, key=self.conclusion_label.count), time_now)
                        message = Mail(
                            from_email='taib1606838@student.ctu.edu.vn',
                            to_emails="nhattaibackup001@gmail.com",
                            subject='NOTIFICATION FROM TRAFFIC DENSITY ESTIMATION SYSTEM',
                            html_content=email_content)
                        try:
                            if attachedFile:
                                message.attachment = attachedFile
                            print("aaaaa", sendgrid_api_key)
                            sg = SendGridAPIClient(sendgrid_api_key)
                            response = sg.send(message)
                            print(response.status_code)
                            print(response.body)
                            print(response.headers)
                            print("Sent email!")
                        except Exception as e:
                            print(e)
                            print("Email sending failed!")
                            pass
                
                # Clear conclusion_label list
                self.conclusion_label.clear

            cv2.putText(frame, label_count, (100, 690), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)             

            # Draw counter area
            frame = cv2.polylines(frame,[np.array([list(p) for p in self.points], np.int32)],True,(255, 0, 0),2)       
            
            # Scale 75% display
            frame = self.rescale_frame(frame)
            cv2.imshow("Traffic Density Estimation", frame)

            if frame_count % self.process_every_x_frame == 0:
                # Save to logs (database)
                cv2.imwrite(img_path, frame)
                print("saved in:",img_path)
                cv2.waitKey(0)
            key = cv2.waitKey(110) & 0xFF
            if key == 27:     
                frame = cv2.putText(frame, "Quit", (int(frame.shape[1]/2) - 20, int(frame.shape[0]/2) - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)                        
                cv2.imshow("Traffic Density Estimation", frame)
                # while len(self.threads)>0 and not self.get_result_t.is_alive():
                #     self.get_result_t.run()
                #     print("Wait for kill thread!")
                #     sleep(3)
                # print("len threads: ",len(self.threads))
                print("Bye!")     
                cv2.destroyAllWindows()           
                break

            if frame_count==500:
                break

            frame_count +=1

        self.vid2.release()
        cv2.destroyAllWindows()


if __name__=='__main__':    
    predict_video = Predict_video()
    # predict_video.set_up_scenario(10,6,20,50,60,15,0) #CAM5
    # predict_video.set_up_scenario(10,6,20,50,60,0,0) #CAM1
    predict_video.set_up_scenario(10,6,20,50,50,0,1) #CAM3
    # predict_video.set_up_scenario()
    video_path = input("Enter path to video file:")
    # predict_video.init_video(video_path, 30, 10, 10, 35 , bbox_threshold=0.6) #CAM5
    # predict_video.init_video(video_path, 14, 1, 1, 50 , bbox_threshold=0.6) #CAM1
    predict_video.init_video(video_path, bbox_threshold=0.6) #CAM3
    if predict_video.get_counter_area()!= False:
        predict_video.run_predict_video()