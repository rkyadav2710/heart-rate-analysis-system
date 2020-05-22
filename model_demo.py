from keras.models import model_from_json
from keras.models import load_model
import cv2
import numpy as np


with open('Model_latest.json', 'r') as f:
	model = model_from_json(f.read())

model.save('Model_latest.h5')


lowerface = np.array([0, 61, 53])
upperface = np.array([180, 208, 226])

cap = cv2.VideoCapture('HR_Train_data/p69v1source1.avi')
Ret, frame_prev = cap.read()

try:
	frame_prev = cv2.resize(frame_prev, (72, 72))
except Exception as e:
	print(str(e))

normalized = np.zeros_like(frame_prev)
while(True):
	ret, image = cap.read()

	if ret == True: 
		Image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		Image = cv2.resize(Image, (72, 72))
		mask = cv2.inRange(Image, lowerface, upperface)	
		res = cv2.bitwise_and(Image, Image, mask=mask)	
		
		diff = res - 1.06*frame_prev
		Sum = res + 1.06*frame_prev
		normalized = cv2.divide(Sum, diff, normalized, 1.25)
		cv2.imshow('frames', image)		
		res = res.reshape(1, 72, 72, 3)
		normalized = normalized.reshape(1, 72, 72, 3)
		print((model.predict([res, normalized])))
		
		frame_prev = res
	else:
		break

	if cv2.waitKey(27)&0xff==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

