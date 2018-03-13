
from os import listdir
import numpy as np
from PIL import Image


# Load data

def retrieve_data(dir_name):
    dataset = []
    labels= {"cat":1,"dog":0}

    height=[]
    width=[]

    for file in listdir(dir_name):

        if file.endswith(".jpg"):
            # Load the image
            img = Image.open(dir_name+"/"+file)
            print(img.mode)
            # Extract the class of the photo
            class_name = file.split(".")


            #************ Preprocessing *************#
            # Preprocessing 1: resize the images (a simple preprocessing)

            img = img.resize((32,32))

            # preprocessing 2: convert the rgb matrix into a column vector
            img_array = np.array(img)

            #print ("Array shape :",img_array.shape)
            img_array = img_array.reshape(img_array.shape[0]*img_array.shape[1]*3,1)
            #print("max and min values : ", np.amax(img_array), np.amin(img_array))

            # preprocessing 3: normalize the vector (Easiest approach just divide by 255 which is the color channel maximum value)
            img_array = img_array/255

            #print("after reshaping : ", img_array.shape)
            #************ Add the image to the dataset *************#

            dataset.append([img_array, labels[class_name[0]]])


    return dataset


def load_dataset():
    train_dataset = retrieve_data("./train")
    test_dataset = retrieve_data("./test")

    train = np.array(train_dataset)
    test = np.array(test_dataset)

    X_train = np.hstack(train[:,0])
    Y_train = train[:,1]
    Y_train = Y_train.reshape(Y_train.shape[0],1)

    X_test =  np.hstack(test[:,0])
    Y_test = test[:,1]
    Y_test = Y_test.reshape(Y_test.shape[0], 1)

    return (X_train, Y_train, X_test , Y_test, {"cat":1,"dog":0})


if __name__=="__main__":
    load_dataset()