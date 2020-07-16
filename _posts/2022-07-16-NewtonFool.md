---
layout: post
title:  "NewtonFool"
author: cvappstore
categories: [CV]
tags: [Adversarial Attack, NewtonFool]
image: https://pbs.twimg.com/profile_images/563716194323025921/OthWJnik_400x400.png
description: 'An adversarial attack on deep learning image classification models'
featured: false
hidden: false
---

[Overview](#overview) | [Limitations](#Limitations) | [Performance](#Performance) | [Feedback](#Feedback)

# Overview

The attack analyzed in this card aims to fool deep neural networks into misclassification by carefully constructing an impercivable change on the input. As an example, we start with an image of a panda, which our neural network correctly recognizes as a “panda” with 57.7% confidence. Add a little bit of carefully constructed noise and the same neural network now thinks this is an image of a gibbon with 99.3% confidence! This is, clearly, an optical illusion — but for the neural network. You and I can clearly tell that both the images look like pandas — in fact, we can’t even tell that some noise has been added to the original image to construct the adversarial example on the right!

<img width="100%" src="https://miro.medium.com/max/1400/1*PmCgcjO3sr3CPPaCpy5Fgw.png" alt="sample image" />

#### Attack Description

Newton fool works with a very simple rationale, finding the shortest distance possible to nudge the classification of an image from the original class, to another class. Therefore, NewtonFool is an untargeted attack, as we do not control which is class to misclassify the image on.

**Input**: Image

**Output**: The original class, adversarial class, and adversarial image

#### Performance

Due to the nature of the algorithm, NewtonFool will always succeed. However, for realistic purposes, we set a X number of iteration and if the adversarial image is still not able to misclassify, we declare the attack as a failure.

# Limitations

As mentioned previously, NewtonFool is an untargeted attack. In real world scenario, this might not be as useful when we require the misclassification to be of a certain class.

# Feedback

We’d love your feedback on the information presented in this card and/or the framework we’re exploring here for model reporting. Please also share any unexpected results. Your feedback will be shared with model and service owners. Contact me @ elimwoon@dsta.gov.sg to provide feedback.