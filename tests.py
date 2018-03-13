import unittest
import img_util
import Logistic_Regression_NN as lr
import numpy as np

class test_cases(unittest.TestCase):

    # Checking img_util.load_dataset() data arrays shapes
    def test_arrays_shapes(self):
        X,Y,XT,YT,C = img_util.load_dataset()
        print("Training data array shape: ", X.shape, "\nTesting data array shape: ", XT.shape, "\nTraining labels vector shape: ", Y.shape,"\nTesting labels vector shape: ",YT.shape)

    # compute_accuracy(true_labels,predicted_labels) test case

    def test_compute_accuracy(self):
        self.assertEqual(lr.compute_accuracy(np.array([0,0,0,0,0,0,0,0,0,1]), np.array([0,0,0,0,0,0,0,0,0,1])),1)
        self.assertEqual(lr.compute_accuracy(np.array([0,0,0,0,0,1,1,1,1,1]), np.array([0,0,0,0,0,0,0,0,0,0])),0.5)

    def test_predict(self):
        self.assertTrue(np.array_equal(lr.predict(np.array([1,2]), 1, np.array([[1,1],[2,2]])),np.array([1,1])))

    def test_segmoid(self):
        self.assertEqual(lr.segmoid(0),0.5)

    def test_estimation(self):
        w, b, cost  = lr.estimate(np.array([0,0]).reshape(2,1), 0, np.array([[1,2],[1,2]]), np.array([1,0]).reshape(2,1), 1.0, 1)
        self.assertTrue(np.array_equal(w,np.array([-0.25, -0.25]).reshape(2,1)))
        self.assertTrue(np.array_equal(cost, np.array([0.6931471805599453])))
        self.assertEqual(b,0)


if __name__ == "__main__":
    unittest.main()
