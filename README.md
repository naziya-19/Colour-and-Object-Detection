Colour and Object Detection
======================
Introduction
----
The main idea of this project is to create a program that can detect colours in an image to the closest precision possible and give an audio output of the colour detected. Secondly it can also detect some basic day to day life objects.

Colour Detection
--------------------------
Colour detection is the process of detecting the name of any colour. Simple isn’t it? Well, for humans this is an extremely easy task but for computers, it is not straightforward. Human eyes and brains work together to translate light into colour. Light receptors that are present in our eyes transmit the signal to the brain. Our brain then recognizes the colour. Since childhood, we have mapped certain lights with their colour names. We will be using the somewhat same strategy to detect colour names.

The Dataset
Colours are made up of 3 primary colours; red, green, and blue. In computers, we define each colour value within a range of 0 to 255. So in how many ways we can define a colour? The answer is 256*256*256 = 16,581,375. There are approximately 16.5 million different ways to represent a colour. In our dataset, we need to map each colour’s values with their corresponding names. But don’t worry, we don’t need to map all the values. We will be using a dataset that contains RGB values with their corresponding names. The CSV file for our dataset has been taken from this link:

Colours Dataset
The wikipedia_color_names.csv file includes 1299 colour names along with their RGB and hex values.

Libraries used
In the following project OpenCV, pandas and Google Text-to-Speech Libraries are used. 
Opencv:  OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products. It returns a three-dimensional array of images read using which we can edit it.

Pandas: Pandas is an open-source Python package that is most widely used for data science/data analysis and machine learning tasks. It is built on top of another package named Numpy, which provides support for multi-dimensional arrays.

Google Text-to-Speech: gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with Google Translate’s text-to-speech API. Writes spoken mp3 data to a file, a file-like object (byte string) for further audio manipulation, or stdout.

Object Detection
------------------------------
Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Well-researched domains of object detection include face detection and pedestrian detection. Object detection has applications in many areas of computer vision, including image retrieval and video surveillance.

In this project, we have created a program using the OpenCV-Python using YOLOv3 and dnn.

DNN
Deep neural networks. A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers.

YOLO
YOLO divides the image into a 13×13 grid of cells. The size of these 169 cells vary depending on the size of the input. For a 416×416 input size that we used in our experiments, the cell size was 32×32. Each cell is then responsible for predicting a number of boxes in the image.

For each bounding box, the network also predicts the confidence that the bounding box actually encloses an object, and the probability of the enclosed object being a particular class.

Most of these bounding boxes are eliminated because their confidence is low or because they are enclosing the same object as another bounding box with very high confidence score. This technique is called non-maximum suppression.
