from numpy import loadtxt
from keras.models import load_model
import numpy as np
import cv2

model = load_model('model.h5')
model.summary()

cap = cv2.VideoCapture(0)
ret, frame_prev = cap.read()

frame_prev = cv2.resize(frame_prev, (160, 160))
normalized = np.zeros((160, 160), dtype=None)

while(cap.isOpened()): 
	ret, frame = cap.read() 
	if ret == True: 
			
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		frame = cv2.resize(frame, (160, 160))

		mask = cv2.inRange(frame, lowerface, upperface)	
		res = cv2.bitwise_and(frame, frame, mask=mask)
		diff = res - 1.06*frame_prev
		Sum = res + 1.06*frame_prev
		normalized = cv2.divide(Sum, diff, normalized, 1.25)
		prediction = model.predict(normalized)
		print(prediction)

	else:
		break

	if(cv2.waitKey(27)&0xff==ord('q')):
		break

cap.release()
cv2.destroyAllWindows()	
