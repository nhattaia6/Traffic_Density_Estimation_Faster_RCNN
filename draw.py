import cv2 
from time import sleep
import numpy as np
import copy

class Get_counter_area():
    def __init__(self, img):
        """
        Input image (cv2.imread())
        """
        self.img_original = img
        # variables 
        self.points = []
        self.center = []
        self.event_list = []
        self.img = copy.deepcopy(self.img_original)

    def process_mouse_click_event(self, event, x, y, flags, param):
        """
        Function to process mouse click event
        if L-click -> point on the counter area
        if R-click -> point inside the counter area
        """
        if event == cv2.EVENT_LBUTTONUP: 
            print("(xp:yp)=({}:{})".format(x, y))
            self.points.append((x,y))
            cv2.circle(self.img, (x, y), 4, (0, 255, 0), -1)
            if len(self.points)>1:
                cv2.line(self.img, pt1 =self.points[-2], 
                        pt2 =self.points[-1], 
                        color =(0, 255, 255),
                        thickness=2
                        )
            cv2.imshow("Set up counter area", self.img) 
            self.event_list.append(0)
        elif event == cv2.EVENT_RBUTTONUP and len(self.center)==0:
            print("(xc:yc)=({}:{})".format(x, y))
            self.center = [x,y]
            cv2.circle(self.img, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("Set up counter area", self.img) 
            self.event_list.append(1)
    
    def set_up_mouse_call_back_event(self):
        """
        Link mouse click with call back event
        """
        cv2.namedWindow(winname = "Set up counter area") 
        cv2.setMouseCallback("Set up counter area",  
                            self.process_mouse_click_event) 
    def run(self):
        """
        Begin draw counter area
        Press Backspace to delete recent point
        Press Enter to finish drawing 
        Return list tuple are points on the area and a tuple is a point inside the area!
        """
        hint1 = 'Press Backspace to delete recent point'
        hint2 = 'Press Enter to finish drawing'
        hint3 = 'L-click to draw a point on the counter area'
        hint4 = 'R-click to draw a point inside the counter area'
        while True: 
            cv2.putText(self.img, hint3, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            cv2.putText(self.img, hint4, (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            cv2.putText(self.img, hint2, (10, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            cv2.putText(self.img, hint1, (10, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Set up counter area", self.img) 
            if cv2.waitKey(5) & 0xFF == 13:
                print("enter")
                if len(self.points)<3 or len(self.center)==0:
                    print("Please! Draw 3 points at least for area and 1 point inside that area!")
                    continue
                cv2.line(self.img, pt1 =self.points[0], 
                            pt2 =self.points[-1], 
                            color =(0, 255, 255),
                            thickness=2
                            )
                print("draw", self.points[0], self.points[-1])
                cv2.imshow("Set up counter area", self.img) 
                cv2.waitKey(0)
                break
            elif cv2.waitKey(5) & 0xFF == 8:
                print("backspace")
                if len(self.event_list)>0:
                    last_event = self.event_list.pop(-1)
                    if last_event==0:
                        if len(self.points)<1:
                            continue
                        self.points = self.points[:-1]
                        # if len(self.points)<2:
                        #     continue
                        self.img = copy.deepcopy(self.img_original)
                        for point in self.points:
                            cv2.circle(self.img, point, 4, (0, 255, 0), -1)
                        cv2.polylines(self.img,[np.array([list(p) for p in self.points], np.int32)],False,(0, 255, 255),2)
                        if len(self.center)>0:
                            cv2.circle(self.img, (self.center[0], self.center[1]), 4, (0, 0, 255), -1)
                    elif last_event==1:
                        self.center = []
                        self.img = copy.deepcopy(self.img_original)
                        for point in self.points:
                            cv2.circle(self.img, point, 4, (0, 255, 0), -1)
                        cv2.polylines(self.img,[np.array([list(p) for p in self.points], np.int32)],False,(0, 255, 255),2)
                cv2.imshow("Set up counter area", self.img)
                cv2.waitKey(100)
            elif cv2.waitKey(5) & 0xFF == ord('q'):
                print("Bye!")
                cv2.destroyAllWindows()
                return False, False
        cv2.destroyAllWindows()
        return self.points, tuple(self.center)

if __name__ =='__main__':
    img = cv2.imread("F:\LVTN\DATA 3\Img_cut_video12_16/video12_16/video12B_001.jpg")
    gca = Get_counter_area(img) 
    gca.set_up_mouse_call_back_event()
    points, center = gca.run()
    print("points:{}, center:{}".format(points, center))
