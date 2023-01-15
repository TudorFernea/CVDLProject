import os
import cv2

test_data_path = "C:/Users/Asus/Desktop/CVDLProject/GTSRB/Test"

for folder in os.listdir(test_data_path):
    for filename in os.listdir(os.path.join(test_data_path, folder)):
        os.chdir(os.path.join(test_data_path, folder))
        img = cv2.imread(os.path.join(test_data_path, folder, filename))
        # print(filename.split(".")[0]+".png")
        if img is not None:
            cv2.imwrite(folder+filename.split(".")[0] + ".png", img)
            os.remove(os.path.join(test_data_path, folder, filename))
