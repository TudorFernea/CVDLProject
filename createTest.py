import os

import cv2
import pandas as pd


df = pd.read_csv("./GTSRB/Test2.csv")
test_data_path = "C:/Users/Asus/Desktop/CVDLProject/GTSRB/Test"

""""
for i in range(0,43):
    path = os.path.join(test_data_path, str(i))
    os.mkdir(path)
"""

for filename in os.listdir(test_data_path):
    if len(filename.split("."))>1:
        cls = None
        for row in df.iloc:
            if row.Path == "Test/"+filename:
                cls = row.ClassId
                break

        os.chdir(os.path.join(test_data_path, str(cls)))
        img = cv2.imread(os.path.join(test_data_path, filename))

        # print(filename.split(".")[0]+".png")
        if img is not None:
            cv2.imwrite(filename, img)
            os.remove(os.path.join(test_data_path, filename))
