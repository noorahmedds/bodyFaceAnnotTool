import cv2
import numpy as np

class AudienceDetector:
    def __init__(self, initial_frame, movement_threshold=500, capacity=5, pixel_threshold=32):
        self.previous_frame = cv2.cvtColor(initial_frame.copy(), cv2.COLOR_BGR2GRAY)
        
        self.old_grays = [cv2.GaussianBlur(self.previous_frame, (5, 5), 0)]
        self.pixel_threshold = pixel_threshold
        self.movement_threshold = movement_threshold
        self.capacity = capacity
        # self.valid_counts = [0] * len(movement_threshold)
        # self.discarded_counts = [0] * len(movement_threshold)
        # self.total_frame_count = 0
        

    def is_audience_present(self, frame):
        # Returns true is audience was detected
        first_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)

        # Applying a 5x5 gaussian filter to blue out the previous image
        first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)    

        second_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.GaussianBlur(second_gray, (5, 5), 0)

        difference = cv2.absdiff(first_gray, second_gray)
        self.previous_frame = frame

        # Caluclating the amount of on pixels
        pixel_threshold = 32
        movement_threshold = 10 #should be a percentage of pixels which are on in the frame
        pixel_on_count = (difference[difference > pixel_threshold]).size

        if np.amax(difference) > movement_threshold:
            # print(np.amax(difference))
            
            # Debugging
            # _, difference2 = cv2.threshold(difference, 13, 255, cv2.THRESH_BINARY)
            # cv2.imshow("difference", difference2)
            
            return True
        else:
            # Debugging
            cv2.imshow('difference', frame)
            return False

    def is_audience_present2(self, frame):
        # first_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # import pdb; pdb.set_trace()

        average_difference = np.zeros_like(gray_frame)
        for old in self.old_grays:
            difference = cv2.absdiff(old, gray_frame)
            average_difference += difference

        average_difference = average_difference / float(len(self.old_grays))
        average_difference = average_difference.astype('uint8')


        # ==> Replace old gray with queue
        # Append the new gray frame to the old grays list
        if len(self.old_grays) == self.capacity:
            self.old_grays.pop(0)
        self.old_grays.append(gray_frame)

        pixel_on_count = (average_difference[average_difference > self.pixel_threshold]).size
        if pixel_on_count > self.movement_threshold:
            # res = cv2.resize(average_difference, dsize = (int(average_difference.shape[1]/2), int(average_difference.shape[0]/2)))
            # cv2.imshow(f"{i}, difference", res)
            return True
        else:
            # res = cv2.resize(frame, dsize = (int(frame.shape[1]/2), int(frame.shape[0]/2)))
            # cv2.imshow(f"{i}, difference", res)
            return False