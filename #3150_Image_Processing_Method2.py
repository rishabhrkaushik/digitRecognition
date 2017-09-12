import numpy as np
import cv2

def ocrDigit(crop, matrix):

	No_pos_D1 = []			#Initialize list to contain data from matrix 1
	No_pos_D2 = []			#Initialize list to contain data from matrix 1
	tempResult = "" 		#String to hold result of detected number
	centNumberD = []		#Initialize a list containing contours of digit, and co-ordinate of thier centNumberD

	binary = cv2.threshold(crop, binary_threshold, 255, cv2.THRESH_BINARY)[1]		#Convert image to binary

	contours = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]	#detect all contours in given cell
	
	for c in contours:																#Filter contours and delete false detected contours
		if(cv2.contourArea(c) > min_area and cv2.arcLength(c,True) < max_perimeter):#Check area and perimeter parameters if satisfied or not														
			M = cv2.moments(c)														#Moments for calculating centNumberD
			cx = int(M['m10']/M['m00'])												#X co-ordinate of centNumberD
			cy = int(M['m01']/M['m00'])												#Y co-ordinate of centNumberD
			centNumberD.append([cx, cy, c])											#Append centNumberD co-ordinates and contout to list

	centNumberD.sort(big)																#Sort this list based on x co-ordinate of centNumberD of contours
										
	if(matrix == 1):
		for i in range(0, len(centNumberD)):											#Itterate over the sorted list
			[x, y, w, h] = cv2.boundingRect(centNumberD[i][2])							#Get bounding rectangle of contour
			number = binary[y:y+h, x:x+w]												#Crop only rectangle bounding contour
			number_small = cv2.resize(number, (20, 20))									#Scale it to 20 * 20 matrix
			number_small = number_small.reshape((1,400))								#Shape it tp 1 * 400 matrix
			number_small = np.float32(number_small)										#Convert it to float32
			retval, results, neigh_resp, dists = model.find_nearest(number_small, k = 1)#Compare this matrix with data initialy saved while training
			No_pos_D1.append(int(results[0][0]))										#Append number detected to list
		return No_pos_D1

	elif(matrix == 2):
		for i in range(0, ((len(centNumberD))/2)):											#Itterate over the sorted list
			for j in range(0, 2):															#Co-catenate two numbers of list
				[x, y, w, h] = cv2.boundingRect(centNumberD[2*i+j][2])						#Get bounding rectangle of contour
				number = binary[y:y+h, x:x+w]												#Crop only rectangle bounding contour
				number_small = cv2.resize(number, (20, 20))									#Scale it to 20 * 20 matrix
				number_small = number_small.reshape((1,400))								#Shape it tp 1 * 400 matrix
				number_small = np.float32(number_small)										#Convert it to float32
				retval, results, neigh_resp, dists = model.find_nearest(number_small, k = 1)#Compare this matrix with data initialy saved while training
				tempResult = str(tempResult) + str(int(results[0][0]))						#Append detected digit with one detected earlier if any
				tempResult = int(tempResult)												#Convert complete number back to string
			No_pos_D2.append((centNumberD[2*i+j][0], centNumberD[2*i+j][1], tempResult)) 	#Append X and Y co-ordinate and contour along with contour to list
			tempResult = ""																	#Reinitialize tempResult
		for i in range(0, len(No_pos_D2)):													#Itterrate over Second matrix list
			No_pos_D2[i] = detCelNumber(No_pos_D2[i])										#Calculate cell number and reassign cell number and number detected 
		return No_pos_D2

def displayContours(image):

	imageContoursFinal = [] 	#final list containing all contours after deleting false detections

	#Convert image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)									#image in gray
	imageBinary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)[1]	#binary image

	#Find Contours
	imageContours = cv2.findContours(imageBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]	#list of all contours

	#iterate over contour list and delete flase contour
	for c in imageContours:																#Loop to delete false detected contours
		if(cv2.contourArea(c) > min_area and cv2.arcLength(c,True) < max_perimeter):	#Select contour only if it fits area and perimeter parameters
			imageContoursFinal.append(c)												#append contour if it pass it fits given parameters

	cv2.drawContours(image, imageContoursFinal, -1, (0, 255, 0), 2)						#draw all contours from imageContoursFinal

	#Show image with plotted contours
	cv2.imshow('ocr', image)	#show image with contours
	cv2.waitKey(0)				#wait till a key is pressed
	cv2.destroyAllWindows()		#destroy all windows

def train(trainImage, sampleName, responseName):
	cnt = []
	gray = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)												#Convert train image to gray
	thresh = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)[1]						#Convert train image to binary

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)			#detect Contours
	samples =  np.empty((0, 400))																	#Create empty array to contain training data
	responses = []																					#Create empty list to contain responses

	keys = [i for i in range(1048624, 1048634)]														#Create a list of keys, the number are those sent my keyboard, may need to recalibrate them
	#keys = [i for i in range(48, 58)]																#Commented since 0 corresponds to 1048624 and 9 to 1048634 on my num pad

	for contour in contours:																		#itterate over contours
	    if(cv2.contourArea(contour) > min_area and cv2.arcLength(contour,True) < max_perimeter):	#if contour big enough
	    	cnt.append(contour)

	for contour in cnt:
		[x, y, w, h] = cv2.boundingRect(contour)													#Get bounding rectangle parameters
		cv2.rectangle(trainImage, (x, y), (x+w, y+h), (0, 0, 255), 2)								#draw bounding rectangle
		number = thresh[y:y+h, x:x+w]																#crop only the contour
		number_small = cv2.resize(number, (20, 20))													#resize the image to 20 * 20

		# show image and wait for keypress
		cv2.imshow('Image', trainImage)																#Display image with bounding rectangle							
		key = cv2.waitKey(0)																		#Wait for a key to be pressed
		#print key 																					#Display the key number(used to recaliibrate keyboard mapping)

		if key == 27:																				#if key is escape key
		    sys.exit()																				#exit program
		elif key in keys:																			#else if key is in numberic range
		    sample = number_small.reshape((1,400))													#re_shape image to 1 * 400
		    samples = np.append(samples,sample,0)													#append it to sample array
		    responses.append(int(chr(key-1048576)))													#append the key pressed to response error
		    print int(chr(key-1048576))																#Print the key pressed

	print "training complete"
	np.savetxt(sampleName + '.data', samples)														#Save the samples
	responses = np.array(responses, np.float32)														#convert responses to float32
	responses = responses.reshape((responses.size,1))												#reshape responses
	np.savetxt(responseName + '.data', responses)													#Save the responses
	print str("Data saved as " + sampleName + '.data' + " and " + responseName + '.data')
	cv2.destroyAllWindows()		#destroy all windows


def big(no1, no2):													#Function to compare two co-ordinates
	if((no2[1] - 10) < no1[1] < (no2[1] + 10)):
		if(no1[0] > (no2[0] + 10)):
			return 1
		elif(no1[0] < (no2[0] + 10)):
			return -1

	elif(no1[1] > (no2[1] + 10)):
		return 1

	elif(no1[1] < (no2[1] - 10)):
		return -1

	else:
		return 0

def detCelNumber(list):																											#Function to determine cell number based on centroid itteration and approximate matching of current centroid
	Ccx, Ccy = 70, 110																											#X and Y co-ordinate of centoid of first cell
	Cw, Ch = 112, 113																											#Cell width and cell height
	for row in range(1, 7):																										#itterate over rows
		for column in range(1, 5):																								#itterate over columns
			if((((Cw * column) - 25) < list[0] < ((Cw * column) + 25)) and (((Ch * row) - 25) < list[1] < ((Ch * row) + 25))): 	#Check if the centroid of contour is near about to centroid of cell detected
				cell_number = ((row-1)*4) + (column-1)																			#Calculate cell number based on rows and columns
				return (cell_number, list[2])	
				break

def play(img):

	#Initialize blank matrices containing result
	No_pos_D1 = []	#Blank list containing all numbers detected from matrix 1
	No_pos_D2 = []	#Blank list containing list of cell number and numbers detected from matrix 2
    
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	#image in gray

	h, w= gray.shape 								#get height and width of image

	left_matrix = gray[:, :(w/2.5)] 				#Left matrix crop
	right_matrix = gray[:, (w/2.1):] 				#Right matrix crop

	No_pos_D1 = ocrDigit(left_matrix, 1) 			#send left matrix to ocrDigit function
	No_pos_D2 = ocrDigit(right_matrix, 2)

	return No_pos_D1, No_pos_D2	


if __name__ == "__main__":
	#Define Constants required ahead
	min_area = 600				#minimum area to prevent false detection on contours
	max_perimeter = 400			#maximum perimeter to prevent false detection on contours
	binary_threshold = 10		#threshold for converting gray image to binary

	#Uncomment follwing section and change parameters to retrain the model
	"""trainImage = cv2.imread('test_image1.jpg')							#Open training image change if necesarry
	if(trainImage is not None):
		train(trainImage, "samples", "responses")							#Change samples and responses for changing file name of saved file
	else:
		print "Train image not loaded, check if it exists and check file name and path"
	"""
    #Train Knn
	samples = np.loadtxt('samples.data', np.float32)					#load sample data set created while training
	responses = np.loadtxt('responses.data', np.float32)				#load response set created while training
	responses = responses.reshape((responses.size,1))					#reshape responses
	model = cv2.KNearest()												#rename cv2.KNearest as model
	model.train(samples, responses)										#train model

	#Check for image
	img = cv2.imread('test_image1.jpg')										#Load image
	if(img is not None):
		No_pos_D1,No_pos_D2 = play(img)										#pass it to detect numbers
		print No_pos_D1														#print first list
		print No_pos_D2														#print second list
		displayContours(img)												#pass image to display number contours
	else: 
		print "Image not loaded, check if it exists and check file name and path"