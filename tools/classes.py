import os
import cv2
import copy
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from . import image_tools as imtool


np.random.seed(286)


class Estimator:
    def __init__(self):
        pass


    def calculate_normalized_distances(self, estimations, num_dims=2):
        self.distance_pairs = ((1, 2), (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (2, 8), 
                               (2, 7), (1, 7), (1, 8), (8, 10), (10, 12), (7, 9), (9, 11))

        p1_list, p2_list = zip(*self.distance_pairs)

        estimations = estimations[:, :, :num_dims]
        distance_list = []
        base_distance = 0

        for person in range(len(estimations)):
            point1 = estimations[:, p1_list]
            point2 = estimations[:, p2_list]
            
            person_distances = (((abs(point1 - point2)) ** 2).sum(axis=2)) ** 0.5
            distance_list.append(person_distances)
        
        distance_list = np.array(distance_list)
        self.distances = distance_list / distance_list[:, :, base_distance:base_distance+1]

        return self.distances


class MovenetEstimator(Estimator):
    def __init__(self, num_people=1, in_width=256, in_height=256):
        self.num_people = num_people
        self.in_width = in_width
        self.in_height = in_height
        self.points = [0] + list(range(5, 17))

        try:
            self.model_path = 'D:/UWE_DS_Materials/CSCT/Codes/weights/movenet'
            self.module = tf.saved_model.load(self.model_path)
        except:
            self.model_path = "https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1"
            self.module = hub.load(self.model_path)
    
        self.model = self.module.signatures['serving_default']

    def __reorder_result__(self):
        self.result = self.result['output_0'].numpy()[:, :self.num_people, :51].reshape((-1,17,3))
        self.result = self.result[:, self.points, :]

    def estimate(self, frame):
        self.frame = frame
        self.frame = imtool.pad_image(self.frame)
        
        self.input_image = tf.expand_dims(self.frame, axis=0)
        self.input_image = tf.image.resize_with_pad(self.input_image, self.in_height, self.in_width)
        self.input_image = tf.cast(self.input_image, dtype=tf.int32)
        
        self.result = self.model(self.input_image)
        self.__reorder_result__()
        
        return self.result


# class OpenCVEstimator(Estimator):
#     def __init__(self, in_width=368, in_height=368):
#         self.in_width = in_width
#         self.in_height = in_height
#         self.model_path = './weights/openCV/graph_opt.pb'
#         self.model = cv2.dnn.readNetFromTensorflow(self.model_path)

#     def __reorder_result__(self):
#         self.__use_points__ = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
#         self.result = self.result[:, self.__use_points__, :, :]

#     def __correct_result__(self):
#         person_corrected = []
#         for person in range(self.result.shape[0]):
#             joint_corrected = []
            
#             for joint in range(self.result.shape[1]):
#                 _, conf, _, point = cv2.minMaxLoc(self.result[person, joint, :, :])
                
#                 x = int((self.__frame_width__ * point[0]) / self.result.shape[3])
#                 y = int((self.__frame_height__ * point[1]) / self.result.shape[2])
#                 joint_corrected.append([x, y])

#             person_corrected.append(joint_corrected)
#         self.result = person_corrected

#     def estimate(self, frame):
#         self.__frame_height__, self.__frame_width__, _ = frame.shape
#         self.model.setInput(cv2.dnn.blobFromImage(frame, 1.0, (self.in_width, self.in_height), (0, 0, 0), swapRB=True, crop=False))
#         self.result = self.model.forward()
#         # self.__reorder_result__()
#         self.__correct_result__()
        
#         return self.result



class DataGenerator:
    def __init__(self):
        self.__data = []
        self.estimator = MovenetEstimator()

    
    @property
    def data(self):
        return copy.deepcopy(self.__data)

    
    @data.setter
    def data(self, new_data):
        self.__data = new_data

    
    def add_data(self, name, path, identifier):
        self.__data.append((name, path, identifier))

    
    def __compile(self):
        self.__files = []
        image_extensions =['jpg', 'jpeg', 'png']
        
        for name, path, identifier in self.data:
            for root, dirs, files in os.walk(path):
                tmp_labels = [identifier(file) for file in  files if file.split('.')[-1] in image_extensions]
                tmp_files = [os.path.join(root, file) for file in  files if file.split('.')[-1] in image_extensions]

            self.__files += list(zip(tmp_files, tmp_labels))

        self.__files = np.array(self.__files)


    def __find_pair(self, is_same):
        num = np.random.choice(len(self.__files), 1)[0]
        sample1, id_name1 = self.__files[num]

        if is_same:
            new_list = [[file, id_] for file, id_ in self.__files if ((sample1 != file) and (id_name1 == id_))]
            
        else:
            new_list = [[file, id_] for file, id_ in self.__files if ((sample1 != file) and (id_name1 != id_))]

        num = np.random.choice(len(new_list), 1)[0]
        sample2, id_name2 = new_list[num]

        return (sample1, sample2, int(is_same))
        
    
    def __get_pairs(self, batch_size):
        self.__compile()

        half_batch = batch_size // 2
        remaining = batch_size - half_batch

        same_pairs = [self.__find_pair(True) for i in range(half_batch)]
        diff_pairs = [self.__find_pair(False) for i in range(remaining)]
        self.__pairs = same_pairs + diff_pairs
        np.random.shuffle(self.__pairs)
    

    def forward(self, batch_size=32, shape=(256,256), grayscale=True):
        self.__get_pairs(batch_size)
        image_tensor1 = np.array([])
        image_tensor2 = np.array([])
        labels = np.array([])
        
        for path1, path2, label in self.__pairs:
            if grayscale:
                image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

            else:
                image1 = cv2.imread(path1)
                image2 = cv2.imread(path2)
            
            image1 = self.estimator.estimate(image1)
            image1 = self.estimator.calculate_normalized_distances(image1)
            image1 = image1.reshape(1, -1)
            
            image2 = self.estimator.estimate(image2)
            image2 = self.estimator.calculate_normalized_distances(image2)
            image2 = image2.reshape(1, -1)
            
            if (len(image_tensor1) > 0) and (len(image_tensor2) > 0):
                image_tensor1 = np.concatenate((image_tensor1, image1))
                image_tensor2 = np.concatenate((image_tensor2, image2))

            else:
                image_tensor1 = image1
                image_tensor2 = image2
                
            labels = np.append(labels, label)

        return image_tensor1, image_tensor2, labels












