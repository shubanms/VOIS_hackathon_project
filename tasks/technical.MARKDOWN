# Description
This project uses object detection to identify and classify waste detected in the image with the help of image detection in python
and neural networks.

Neural networks are parallel computing devices, 
which are basically an attempt to make a computer model of the brain.
The main objective is to develop a system to perform various computational tasks faster than the traditional systems.

Artificial intelligence and machine learning haven’t just grabbed headlines and made for blockbuster movies;
they’re poised to make a real difference in our everyday lives, such as with self-driving cars and life-saving medical devices.
In fact, according to Global Big Data Conference, AI is “completely reshaping life sciences, medicine, and healthcare” and is also transforming voice-activated assistants,
image recognition, and many other popular technologies.

Artificial Intelligence is a term used for machines that can interpret the data, learn from it,
and use it to do such tasks that would otherwise be performed by humans.
Machine Learning is a branch of Artificial Intelligence that focuses more on training the machines to learn on their own without much supervision.

# What is a Neural Network?
When you ask your mobile assistant to perform a search for you—say, Google or Siri or Amazon Web—or use a self-driving car,
these are all neural network-driven. Computer games also use neural networks on the back end, as part of the game system and how it adjusts to the players,
and so do map applications, in processing map images and helping you find the quickest way to get to your destination.

A neural network is a system or hardware that is designed to operate like a human brain.

Neural networks can perform the following tasks:

1 Image detection
2 Translate text
3 Identify faces
4 Recognize speech
5 Read handwritten text
6 Control robots
And a lot more

# Working of Neural Network

- A neural network is usually described as having different layers.
The first layer is the input layer, it picks up the input signals and passes them to the next layer.
The next layer does all kinds of calculations and feature extractions—it’s called the hidden layer.
Often, there will be more than one hidden layer. And finally, there’s an output layer, which delivers the final result.

<img src="https://www.simplilearn.com/ice9/free_resources_article_thumb/layers-of-a-neural-network-1.jpg" width = "600" height = "400" >

Let’s take the real-life example of how traffic cameras identify license plates and speeding vehicles on the road. The picture itself is 28 by 28 pixels, and the image is fed as an input to identify the license plate. Each neuron has a number, called activation, which represents the grayscale value of the corresponding pixel, ranging from 0 to 1—it’s 1 for a white pixel and 0 for a black pixel. Each neuron is lit up when its activation is close to 1.

Pixels in the form of arrays are fed into the input layer. If your image is bigger than 28 by 28 pixels, you must shrink it down, because you can’t change the size of the input layer. In our example, we’ll name the inputs as X1, X2, and X3. Each of those represents one of the pixels coming in. The input layer then passes the input to the hidden layer. The interconnections are assigned weights at random. The weights are multiplied with the input signal, and a bias is added to all of them.

<img src = "https://www.simplilearn.com/ice9/free_resources_article_thumb/slide-34-how-does-a-neural-network-work-1.jpg" width = "600" height = "400" >

The weighted sum of the inputs is fed as input to the activation function, to decide which nodes to fire for feature extraction. As the signal flows within the hidden layers, the weighted sum of inputs is calculated and is fed to the activation function in each layer to decide which nodes to fire.

<img src="https://www.simplilearn.com/ice9/free_resources_article_thumb/2-how-does-a-neural-network-work-1.jpg" width = "600" height = "400" >

# Types of activation functions



