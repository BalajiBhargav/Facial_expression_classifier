# Facial_expression_classifier
Dataset - https://www.kaggle.com/msambare/fer2013
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

I have trained Deep Learning model with 4 categories (angry,sad,happy,neutral) with accuracy of 71%

With the help of this model i want to create another project in which using harcascade face detector,can detect the face and send the image which contains this face to our DeepLearning model that detects the facial expression so according to the mood we can perform certain activities(Ex: When you are angry playing some funny videos etc..)
