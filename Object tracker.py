import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_values(bounding_box_1, bounding_box_2):
    x_overlap = max(0, min(bounding_box_1[0] + bounding_box_1[2], bounding_box_2[0] + bounding_box_2[2]) - max(bounding_box_1[0], bounding_box_2[0]))
    y_overlap = max(0, min(bounding_box_1[1] + bounding_box_1[3], bounding_box_2[1] + bounding_box_2[3]) - max(bounding_box_1[1], bounding_box_2[1]))

    intersection_area = x_overlap * y_overlap

    bounding_box_1_area = bounding_box_1[2] * bounding_box_1[3]
    bounding_box_2_area = bounding_box_2[2] * bounding_box_2[3]

    union_area = bounding_box_1_area + bounding_box_2_area - intersection_area
    iou_score = intersection_area / union_area if union_area != 0 else 0.0

    return iou_score

def calculate_precision_recall(list_i, threshold_value, total):
    a = 0
    b = 0
    c = 0
    
    for i in list_i:
        if i >= threshold_value:
            a += 1
        else:
            b += 1
            
    c = total - a
    precision = a / (a + b)
    recall = a / (a + c)

    return precision, recall
    

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[]
 
    if int(minor_ver) < 1:
        tracker = cv2.Tracker_create(tracker_type)
    else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.legacy.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.legacy.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.legacy.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.legacy.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                tracker = cv2.legacy.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                tracker = cv2.legacy.TrackerCSRT_create()
 

    annotation_path = r"skater.txt"
    fp = open(annotation_path,'r')
    content = fp.readlines()

    ground_truth_coordinates = [list(map(float, line.strip().split()[1:])) for line in content]

    video = cv2.VideoCapture("Skater.mp4")
 
    if not video.isOpened():
        print('Could not open video')
        sys.exit()
 
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file') 
        sys.exit()
     
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
 
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    i=0
    iou=[]
    p_list=[]
    r_list=[]

    while True:

        if(i < len(ground_truth_coordinates)):
            gt_frame = ground_truth_coordinates[i]
            x, y, w, h = map(float, gt_frame)
            height, width, _ = frame.shape
            i += 1
            
            x_pixelated = int(x * width)
            y_pixelated = int(y * height)
            w_pixelated = int(w * width)
            h_pixelated = int(h * height)

            x1 = x_pixelated - w_pixelated // 2
            y1 = y_pixelated - h_pixelated // 2

            x2 = x1 + w_pixelated
            y2 = y1 + h_pixelated

            gt_bbox = (x1, y1, w_pixelated, h_pixelated)

        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking/Prediction Boundary Box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
            #Grounding Boundary Box
            g1 = (x1, y1)
            g2 = (x1 + w_pixelated, y1 + h_pixelated)
            cv2.rectangle(frame, g1, g2, (0, 0, 255), 2, 1)

            iou.append(get_values(gt_bbox,bbox))
            p,r = (calculate_precision_recall(iou,0.5,len(ground_truth_coordinates)))
            p_list.append(p)
            r_list.append(r)
            
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    print(np.std(iou))
    print(np.mean(iou))
    print(np.var(iou))