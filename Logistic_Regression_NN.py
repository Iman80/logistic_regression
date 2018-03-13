import img_util
import numpy as np
import pandas as pd

'''
Image classification process using neural networks and logistic regression:
    - Specify the features.
    - Initialize the parameters.
    -  Repeat:
        * Forward propagation.
        * Backward propagation.
        * Optimize using gradient decent (update parameters).
'''


def segmoid(x):
    return 1.0/(1.0+np.exp(-x))


def estimate(w,b,X,Y, alpha,number_of_iterations):
    cost= []
    for i in range(number_of_iterations):
        print(".",end="",flush=True)
        # Forward propagation
        Z = segmoid((np.dot(np.transpose(w),X) + b).astype(float))
        assert(Z.shape == (1,X.shape[1]))
        Z = np.transpose(Z)
        assert (Z.shape == Y.shape)
        # Compute cost function
        if i%100 == 0:
            cost.append(1.0/Y.shape[0] * np.sum(-1.0 * ((Y * np.log(Z))+ ((1-Y)* np.log(1-Z)))))
        # Backward propagation.
        dz = Z - Y
        dw = 1.0/Y.shape[0] * np.dot(X,dz)
        db = 1.0/Y.shape[0]  * np.sum (dz)


        # Update parameters.
        w = w - alpha * dw
        b = b - alpha * db

    return w, b, np.array(cost)

def predict(w,b,data):
    Z = segmoid((np.dot(np.transpose(w), data) + b).astype(float))
    return np.array([0 if x <=0.5 else 1 for x in Z.flatten()])


def evaluate(w,b,data,labels):
    predictions = predict(w, b, data)
    acc = compute_accuracy(labels,predictions)
    print ("The accuracy of the classification is ", acc)

def compute_accuracy(true_labels,predicted_labels):
    return 1 - np.mean(np.abs(true_labels - predicted_labels))

    # print accuracy / area under the curve / plot / precision and recall if needed


if __name__== '__main__':
    training_features,training_labels, testing_features,testing_labels, class_labels = img_util.load_dataset()
    tl = set(training_labels.flatten())
    if len(tl) != 2:
        print("The algorithm is designed for binary predictions, please provide negative examples or make sure that your data has only two classes")
        exit()
    # initialize w and b
    w = np.zeros((training_features.shape[0],1))
    b = 0.0
    w, b , cost = estimate(w, b, training_features, training_labels, 0.01, 1000)

    evaluate(w, b, training_features,training_labels)
    evaluate(w,b,testing_features,testing_labels)






