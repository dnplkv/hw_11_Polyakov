import cv2
import os
import pickle
import logging
from src.parse_args import parse_args, args, folders
from src.face_cascade import face_casc


def param_test(func):
    def wrapper():
        expect_faces = 11
        count = 0
        for key, value in func().items():
            if isinstance(value, list):
                count += len(value)
        if count == expect_faces:
            print(f'[*] {count} of {expect_faces} face(s) was/were found. Test completed!')
        else:
            print(f'[*] {count} of expected {expect_faces} face(s) was/were found! Test failed!')
    return wrapper


def benchmark(iters):
    def actual_decorator(func):
        import time

        def wrapper(*args, **kwargs):
            total = 0
            for i in range(iters):
                start = time.time()
                return_value = func(*args, **kwargs)
                end = time.time()
                total = total + (end - start)
            print('[*] Average time for completing: {} sec.'.format(total / iters))
            return return_value

        return wrapper

    return actual_decorator


@param_test
@benchmark(iters=10)
def get_faces():
    """Finding faces on images in folder and
    adding faces coordinates of specified picture to dictionary
            """
    for file in os.listdir(folders):
        img = cv2.imread(os.path.join(folders, file))
    data = {}
    for file in os.listdir(folders):
        img = cv2.imread(os.path.join(folders, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_casc().detectMultiScale(gray, 1.1, 4)
        data[file] = faces.tolist()
        for (x, y, w, h) in faces:  # ROI  - region of interest
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        with open('faces_dump.pickle', 'wb') as fl:
            """Sending dictionary to the pickle.dump
            """
            pickle.dump(data, fl)
    return data


get_faces()


with open('faces_dump.pickle', 'rb') as fl:
    data_new = pickle.load(fl)

for key, value in data_new.items():
    logging.info(f'{key} : {value}')
    print(key, ' : ', value)



