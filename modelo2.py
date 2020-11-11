# face_trainer.py
import cv2, numpy, os

fn_haar = 'haarcascade_frontalface_default.xml'

fn_dir = 'C:\OficinaTechDay\dadosTreino\fabiano'

# Part 1: Create fisherRecognizer
print('Training...')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(fn_dir):

    for subdir in dirs:

        names[id] = subdir

        subjectpath = os.path.join(fn_dir, subdir)

        for filename in os.listdir(subjectpath):

            path = subjectpath + '/' + filename

        images.append(cv2.imread(path, 0))

        labels.append(subdir)

        id += 1

(im_width, im_height) = (112, 92)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images

model = cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)

model.write('trainer/trainer.yml')