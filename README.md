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

<img src="https://www.simplilearn.com/ice9/free_resources_article_thumb/layers-of-a-neural-network-1.jpg" width = "800" height = "400" >

Let’s take the real-life example of how traffic cameras identify license plates and speeding vehicles on the road. The picture itself is 28 by 28 pixels, and the image is fed as an input to identify the license plate. Each neuron has a number, called activation, which represents the grayscale value of the corresponding pixel, ranging from 0 to 1—it’s 1 for a white pixel and 0 for a black pixel. Each neuron is lit up when its activation is close to 1.

Pixels in the form of arrays are fed into the input layer. If your image is bigger than 28 by 28 pixels, you must shrink it down, because you can’t change the size of the input layer. In our example, we’ll name the inputs as X1, X2, and X3. Each of those represents one of the pixels coming in. The input layer then passes the input to the hidden layer. The interconnections are assigned weights at random. The weights are multiplied with the input signal, and a bias is added to all of them.

<img src = "https://www.simplilearn.com/ice9/free_resources_article_thumb/slide-34-how-does-a-neural-network-work-1.jpg" width = "800" height = "400" >

The weighted sum of the inputs is fed as input to the activation function, to decide which nodes to fire for feature extraction. As the signal flows within the hidden layers, the weighted sum of inputs is calculated and is fed to the activation function in each layer to decide which nodes to fire.

<img src="https://www.simplilearn.com/ice9/free_resources_article_thumb/2-how-does-a-neural-network-work-1.jpg" width = "800" height = "400" >

# Types of activation functions

## 1. Sigmoid Function

The sigmoid function is used when the model is predicting probability.

<img src = "https://www.simplilearn.com/ice9/free_resources_article_thumb/sigmoid-function-neural-networks.jpg" width = "800" height = "400" >

## 2. Threshold Function

The threshold function is used when you don’t want to worry about the uncertainty in the middle.

<img src = "https://www.simplilearn.com/ice9/free_resources_article_thumb/the-threshold-function-1.jpg" width = "800" height = "400" >

## 3. ReLU (rectified linear unit) Function

The ReLU (rectified linear unit) function gives the value but says if it’s over 1, then it will just be 1, and if it’s less than 0, it will just be 0. The ReLU function is most commonly used these days.

<img src = "https://www.simplilearn.com/ice9/free_resources_article_thumb/relu-function-neural-networks.jpg" width = "800" height = "400" >

## 4. Hyperbolic Tangent Function

The hyperbolic tangent function is similar to the sigmoid function but has a range of -1 to 1.

<img src = "https://www.simplilearn.com/ice9/free_resources_article_thumb/hyperbolic-tangent-function-1.jpg" width = "800" height = "400" >


Finally, the model will predict the outcome, applying a suitable application function to the output layer. In our example with the car image, optical character recognition (OCR) is used to convert it into text to identify what’s written on the license plate. In our neural network example, we show only three dots coming in, eight hidden layer nodes, and one output, but there’s really a huge amount of input and output.

<img src = "https://www.simplilearn.com/ice9/free_resources_article_thumb/3-how-does-a-neural-network-work-2.jpg" width = "800" height = "400" >

Error in the output is back-propagated through the network and weights are adjusted to minimize the error rate. This is calculated by a cost function. You keep adjusting the weights until they fit all the different training models you put in.

The output is then compared with the original result, and multiple iterations are done for maximum accuracy. With every iteration, the weight at every interconnection is adjusted based on the error. That math gets complicated, so we’re not going to dive into it here. But, we would look at how it’s being done while executing the code for our use case.

In this project we use a ReLU function to detect images in our neural network.

We have a weights file with all the specified weight's and perform convolution on the image details
before we move on to the next layer.

In this type, the input features are taken in batches—as if they pass through a filter. This allows the network to remember an image in parts. Applications include signal and image processing, such as facial recognition.

This project is done in python specifically Jupyter notebooks using python modules such as NumPy, MatPlotLib and pyplot, CV2