---
layout: post
title:  "Fantastic Hypebeasts and Where to Find Them"
author: kayne_ivan_west
categories: [CV]
tags: [hypebeast, classification]
image: https://loving-lovelace-c9b120.netlify.app/images/blog/hypebeast_example5.jpeg
description: 'Measuring fashion with AI'
featured: false
hidden: false
---

# Hypebeast
21'st century fashion can be confusing. Teenagers queueing for hours outside branded boutiques to buy thousand-dollar kicks (shoes), only to pose for some photos in them before selling them on the internet for a surprisingly handsome profit. Clothes, bags, caps, shovels, and even kayaks spike in value when they are branded with stickers bearing the infamous word “Supreme”. Gram (Instagram) ‘models’ that do not let anything without a brand name touch their body so that they can impress people they do not know with wealth they do not have. These people who thirst to be part of a trend are commonly known as “hypebeast”. But what is hypebeast? Originally a derogatory term for those who favour expensive, hyped-up brands but suffer a debilitating lack of style, it has now lost its negative connotation as it promotes the lack of style as a style itself.

<img src="https://loving-lovelace-c9b120.netlify.com/images/blog/supreme1.png" width="50%" />


Popular instagrammer and self-proclaimed hypebeast Eugene Ang puts it simply: “Hypebeast is a way of life”. However, identifying hypebeast individuals is not so easy as the term is, at best, vague. As such, there was a need to create a tool to identify the style, an impartial digital judge to clear all doubts. In this article, I summarise the project to construct a convolutional neural network and train it to classify hypebeast.

# Collecting the data
For this project, images were collected to use as training data for the hypebeast classifier. Web scrapers were used such as google-images-download and instagram-scraper to collect images from the internet. For google-images-download, keywords such as “hypebeast” were used to find thousands of images on google image search for download. Several famous hypebeast instagram accounts were scraped using instagram-scraper to obtain another few thousand images of fashion commonly agreed to be hypebeast. Negative examples were also obtained for the classifier to train with, and were obtained with google-images-download with keywords such as “office wear”, “normal clothes”, and “stock photo people”. In all, around 26,000 hypebeast images and 7,200 non-hypebeast images were collected.

The collected images were then filtered to keep only pictures that contained humans. For this task, a pretrained ResNet50 with ImageNet weights was used. All images were fed through yolov3, an object detector and image classifier that stand for “you only look once”. Only those with 85% and above confidence of the presence of a human were kept to be used for training and testing. After filtering, around 12,700 hypebeast and 5,300 non- hypebeast images were kept. They were each split in an 8:1:1 ratio to be training, validation, and testing sets respectively.

<img src="https://loving-lovelace-c9b120.netlify.com/images/blog/hypebeast_example.jpeg" />

# Training the model
The images were first passed through a data generator to augment the data slightly and produce a more rounded training set. The training images were subject to shear up to 0.2, zoom up to 0.2, and flipped horizontally. All images were also resized to fit the target size of 224 by 224 pixels, and pixel values were rescaled to range between 0 and 1. The data generator modified the images and fed them into the model for training in batches of size 32.

Transfer learning was applied to create the hypebeast classifier. ResNet50 with ImageNet weights was used as a base model for the classifier. A global average pooling layer and two dense layers with relu activation and a softmax output layer were added behind ResNet50. This resulted in a complex and deep model with more than 24.7 million training parameters. The model was compiled using the rmsprop optimizer on sparse categorical entropy loss and run for 20 epochs on the training set. After that, the original 50 layers of ResNet50 were frozen, and the model was recompiled with the stochastic gradient descent algorithm with a learning rate of 0.0001 and momentum of 0.8. The final few layers were trained a further 10 epochs using this setup to obtain a classifier that was 87.4% accurate at predicting hypebeast images in the validation set.

# Results
The weights for the trained model were saved in a .h5 file to be used for classification. To use the model, we input a query image, and measured the confidence value for the classes “Hypebeast” and “Normal”. The class with the higher confidence value was selected to be the result of the classification, and the result and its confidence were displayed as an output.

## Hypebeast or not?
![](https://loving-lovelace-c9b120.netlify.com/images/blog/hypebeast_example4.png)

![](https://loving-lovelace-c9b120.netlify.com/images/blog/hypebeast_negative2.png)

![](https://loving-lovelace-c9b120.netlify.com/images/blog/lame_cat.png)

![](https://loving-lovelace-c9b120.netlify.com/images/blog/hype_cat.png)

<sup>*work done as an intern at DH</sup>
