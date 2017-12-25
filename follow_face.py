#coding:utf-8
import sys
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np
from threading import Thread
from onvif import ONVIFCamera

test_mode = "onet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['./data/MTCNN_model/PNet_landmark/PNet', './data/MTCNN_model/RNet_landmark/RNet', './data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

#video_capture.set(3, 340)
#video_capture.set(4, 480)

corpbbox = None

W=320
H=160
TH_X = 20 #threshold move x
TH_Y = 20 #threshold move y
SPEED_X = 40
SPEED_Y = 40 
#video_capture = cv2.VideoCapture("rtsp://admin:admin@10.0.1.67:554/0/video1")
src = "rtsp://admin:admin@10.0.1.67:554/0/video1"

class ONVIFCamera:
    def __init__(self,ip,username='admin',password='admin'):
        self.camera = ONVIFCamera(ip, 80, username, password)
        media = self.camera.create_media_service()
        self.ptz = self.camera.create_ptz_service()
        media_profile = media.GetProfiles()[0];
        media_profile.PTZConfiguration._token = media_profile.PTZConfiguration.token
        # Get PTZ configuration options for getting continuous move range
        self.request = self.ptz.create_type('GetConfigurationOptions')
        self.request.ConfigurationToken = media_profile.PTZConfiguration.token
        ptz_configuration_options = self.ptz.GetConfigurationOptions(request)    
        self.request = self.ptz.create_type('ContinuousMove')
        self.request.ProfileToken = media_profile.token    
        self.ptz.Stop({'ProfileToken': media_profile.token})
        
        self.XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
        self.XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
        self.YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
        self.YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min
        
        self.center = (W/2,H/2)

    def set_center(self,center):
        self.tcenter = center
    def follow(self):
        delta_x = self.tcenter[0] - self.center[0]
        delta_y = self.tcenter[1] - self.center[1]
        print("dX : %d, dY %d" % (delta_x,delta_y))
        if abs(delta_x) > TH_X:
            xmove = XMIN if delta_x < 0 else XMAX
            self.center[0] += SPEED_X if delta_x > 0 else -SPEED_X
        else:
            xmove = 0 #
        if abs(delta_y) > TH_Y:
            ymove = YMIN if delta_y < 0 else YMAX
            self.center[1] += SPEED_Y if delta_y > 0 else -SPEED_Y
        else:
            ymove = 0
        print("Move X %d, Move Y %d" %(xmove,ymove))
        #else if not self.is_moving and xmove != 0: #start moving here
        #########
        #self.request.Velocity ={ 'PanTilt' : { 'x':xmove,'y':ymove }}
        #self.ptz.ContinuousMove(self.request)
        #########
        time.sleep(0.4)


class RTSPStream:
    def __init__(self,src):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed,self.frame) = self.stream.read()
        self.stopped = False
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True

vs = RTSPStream(src).start()

center_point = (W/2,H/2)
while True:
    t1 = cv2.getTickCount()
    frame = vs.read()#video_capture.retrieve()
    frame = cv2.resize(frame, (W, H,)) #original 640 320
    if True: #check if grabbed 
        image = np.array(frame)
        boxes_c,landmarks = mtcnn_detector.detect(image)        
        #print landmarks.shape
        t2 = cv2.getTickCount()
        t = (t2 - t1) / cv2.getTickFrequency()
        fps = 1.0 / t
        #if len(boxes_c) == 0:
        #    continue
        xmin = 9999
        xmax = -9999
        ymin = 9999
        ymax = -9999
        found = False
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # if score > thresh:
            x1 = corpbbox[0]
            y1 = corpbbox[1]
            x2 = corpbbox[2]
            y2 = corpbbox[3]
            xmin = x1 if x1 < xmin else xmin
            xmax = x2 if x2 > xmax else xmax
            ymin = y1 if y1 < ymin else ymin
            ymax = y2 if y2 > ymax else ymax
            found = True
	    cv2.rectangle(frame, 
                        (x1, y1), 
                        (x2, y2), 
                        (255, 0, 0), 1)

            cv2.putText(frame, 
                        '{:.3f}'.format(score), 
                        (corpbbox[0], corpbbox[1] - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        #FPS 
        cv2.putText(frame, 
                        '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), 
                        (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2)
        #draw center frame
        if found > 0:
            center_point = (xmin+(xmax - xmin)/2,ymin+(ymax-ymin)/2)
            #cv2.circle(frame,center_point, 2, (0,0,255), -1)
        cv2.line(frame,(center_point[0],0),(center_point[0],H),(0, 0, 255),1)
        cv2.line(frame,(0,center_point[1]),(W,center_point[1]),(0, 0, 255),1)
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i])/2):
                cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))            
        # time end
        cv2.imshow("", frame)
        try:
            cv2.waitKey(1)
        except:
            break
    else:
        print 'device not find'
        break

#video_capture.release()
cv2.destroyAllWindows()
vs.stop()
