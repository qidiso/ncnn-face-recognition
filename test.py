import numpy as np
import facerecognition, cv2

class call_cpp(object):
    def __init__(self, name):
        self.itcom = mobilefacenet.communication(name)

    def run(self, img) :
        ret_list = self.itcom.test(img.shape[0], img.shape[1], image_char)
        gray = np.zeros([img.shape[0], img.shape[1]], np.ubyte)
        c = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                gray[i][j] = ret_list[c]
                c += 1
        return gray

    def run_class(self):
        return self.itcom.ret_box_list()

if __name__ == '__main__':
    facerecog = facerecognition.facerecognition("../models/")

    img = cv2.imread('1.jpg')

    image_char = img.astype(np.uint8).tostring()
    facerecog.recognize(img.shape[0], img.shape[1], image_char)
