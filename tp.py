
import pickle
import numpy as np
import cv2

import os
print("----------------------------------------------------------------")


loaded_model = open("trained_model.p", "rb")
model = pickle.load(loaded_model)

# Specify the directory path
directory = 'images'

# Iterate over files in the directory
for filename in os.listdir(directory):
    text_num = []
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(directory, filename)):
        # Create the full path to the file
        path = os.path.join(directory, filename)
        # Perform actions with the file
        print(path)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        rslt=""
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            digit = th[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            print(padded_digit.shape)
            digit = padded_digit.reshape(1, 28, 28, 1)
            digit = digit / 255.0

            pred = model.predict([digit])[0]
            final_pred = np.argmax(pred)
            text_num.append([x, final_pred])
            rslt+=str(final_pred)


            data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 0)
            thickness = 1
            cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)


        
        text_num = sorted(text_num, key=lambda t: t[0])
        text_num = [i[1] for i in text_num]
        final_text = "".join(map(str, text_num))
        print("result ==== "+str(rslt))
        # cv2.imshow('image', image)
        # cv2.waitKey(0)


