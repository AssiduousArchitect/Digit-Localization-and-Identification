import cv2
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from keras.models import model_from_json

def load_model():
	path = "Models\\"
	json_file = open(path + 'model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	
	model = model_from_json(loaded_model_json)
	model.load_weights(path + "model.h5")
	print("Loaded model from disk")
	return model
	
def predict(test_image):
	test_image = cv2.bitwise_not(test_image)
	test_image_gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(test_image_gray,150,255,cv2.THRESH_BINARY_INV) 

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	dilated = cv2.dilate(thresh,kernel,iterations = 13)
	__, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

	rectangles = [cv2.boundingRect(ctr) for ctr in contours]

	model = load_model()
	for rect in rectangles:
	
		cv2.rectangle(test_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 3)
		leng = abs(int(rect[3]))
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
	
	
		extracted_image = thresh[pt1:pt1+leng, pt2:pt2+leng]
		extracted_image = cv2.resize(extracted_image, (28, 28))
		extracted_image = extracted_image.reshape(1,1,28,28).astype('float32')
	
		prediction = np.argmax(model.predict(extracted_image), axis = 1)
		cv2.putText(test_image, 'Prediction: ' + str(int(prediction[0])), (rect[0], rect[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
	
	cv2.imshow("Predictions", test_image )
	cv2.waitKey(0)
	
drawing = False # true if mouse is pressed
last_x, last_y = -1,-1

# mouse callback function
def mouse_event_handler(event, x, y, flags, param):

    global last_x, last_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (last_x, last_y) ,(x,y), (255,255,255), 5)
            last_x = x
            last_y = y

    elif event == cv2.EVENT_LBUTTONUP: drawing = False

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Draw Digit')
cv2.setMouseCallback('Draw Digit',mouse_event_handler)

while(1):
    cv2.imshow('Draw Digit',img)
    

    if cv2.waitKey(1) & 0xFF == 27:
        predict(img)
        break

cv2.destroyAllWindows()
