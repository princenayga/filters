import face_recognition
import cv2
import os
import numpy as np  
import dlib  
from scipy.spatial import distance as dist  
from scipy.spatial import ConvexHull  

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"  
   
FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
   
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  
   
detector = dlib.get_frontal_face_detector()  
   
predictor = dlib.shape_predictor(PREDICTOR_PATH)  
   
def eye_size(eye):  
	eyeWidth = dist.euclidean(eye[0], eye[3])  
	hull = ConvexHull(eye)  
	eyeCenter = np.mean(eye[hull.vertices, :], axis=0)  
	eyeCenter = eyeCenter.astype(int)  
   
	return int(eyeWidth), eyeCenter  

def place_eye(frame, eyeCenter, eyeSize):  
	eyeSize = int(eyeSize * .5)  
	x1 = int(eyeCenter[0,0] - (eyeSize/2))  
	x2 = int(eyeCenter[0,0] + (eyeSize/2))  
	y1 = int(eyeCenter[0,1] - (eyeSize/2))  
	y2 = int(eyeCenter[0,1] + (eyeSize/2))  
	h, w = frame.shape[:2]  
   
   # check for clipping  
	if x1 < 0:  
		x1 = 0  
	if y1 < 0:  
		y1 = 0  
	if x2 > w:  
		x2 = w  
	if y2 > h:  
		y2 = h  
   
   # re-calculate the size to avoid clipping  
	eyeOverlayWidth = x2 - x1  
	eyeOverlayHeight = y2 - y1  
   
   # calculate the masks for the overlay  
	eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
	mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
	mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   
   # take ROI for the verlay from background, equal to size of the overlay image  
	roi = frame[y1:y2, x1:x2]  
   
   # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.  
	roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
   # roi_fg contains the image pixels of the overlay only where the overlay should be  
	roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)  
   
   # join the roi_bg and roi_fg  
	dst = cv2.add(roi_bg,roi_fg)  
   
   # place the joined image, saved to dst back over the original image  
	frame[y1:y2, x1:x2] = dst  
   
 #---------------------------------------------------------  
 # Load and pre-process the eye-overlay  
 #---------------------------------------------------------  
 # Load the image to be used as our overlay  
imgEye = cv2.imread('./Eye3.png',-1)  
  
 # Create the mask from the overlay image  
orig_mask = imgEye[:,:,3]  
   
 # Create the inverted mask for the overlay image  
orig_mask_inv = cv2.bitwise_not(orig_mask)  
   
 # Convert the overlay image image to BGR  
 # and save the original image size  
imgEye = imgEye[:,:,0:3]  
origEyeHeight, origEyeWidth = imgEye.shape[:2]  

#------------------------------------------------------------------------------------------
class Filters:
	def __init__(self):
		self.imageid = self.filter()



	def filter(self):			
		alpha = 0.4
		beta = 1.0 - alpha
		image = cv2.imread('bg.jpg')
		image = cv2.resize(image, (320, 240))


		stream = cv2.VideoCapture(0)
		stream.set(3, 320)
		stream.set(4, 240)

		print("Press the spacebar to capture image")

		while True:
			count = 1
			imageid = "filteredimage" + str(count) + ".jpg"
			while os.path.exists(imageid):
				count += 1
				imageid = "filteredimage" + str(count) + ".jpg"
			ret, frame = stream.read()
			if ret:  
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				rects = detector(gray, 0)
			for rect in rects:  
				x = rect.left()  
				y = rect.top()  
				x1 = rect.right()  
				y1 = rect.bottom()  
   
				landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  
   
				left_eye = landmarks[LEFT_EYE_POINTS]  
				right_eye = landmarks[RIGHT_EYE_POINTS]  
   
      			 # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)  
   
				leftEyeSize, leftEyeCenter = eye_size(left_eye)  
				rightEyeSize, rightEyeCenter = eye_size(right_eye)  
   
				place_eye(frame, leftEyeCenter, leftEyeSize)  
				place_eye(frame, rightEyeCenter, rightEyeSize)  
   
			#face_landmarks_list = face_recognition.face_landmarks(frame)
				
			fframe = cv2.addWeighted(frame, alpha, image, beta, 0.0)
			cv2.imshow('Video', fframe )
			#print(face_landmarks_list)
			if cv2.waitKey(1) & 0xFF == ord(' '):
				cv2.imwrite(imageid, fframe)
				break

		stream.release()
		cv2.destroyAllWindows()
		return imageid

	def to_open(self):
		to_open = input("Want to view filtered picture? Y?N ")
		if to_open == 'y':
			imageid = self.imageid
			to_open_image = cv2.imread(imageid)
			cv2.imshow(imageid, to_open_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows
		else:
			pass

#-------------------------------------------------------------------------------------------

#image = Filters()
#image.to_open()
