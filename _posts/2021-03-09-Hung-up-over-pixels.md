---
layout: post
title:  "Hung up over pixels"
author: levan
categories: [CV]
tags: [Object Detection]
image: assets/images/hung-up-over-pixels/pixels.svg
description: "Hung up over pixels"
featured: true
hidden: false
---

# Hung up over pixels

## Background

Many tend to ask what is the minimum number of pixels of targets we need for detection, and this is a difficult answer to give in today's world.

## Tradition?

I do believe this tendency of using minimum number of pixels as a metric of comparison, stem from how object detection was traditionally done, using hand-crafted traditional computer vision methods.

## These days

These days with object detectors that are deep learned, gathering specifications on minimum number of pixels of target is moot.

## Bring it to the extreme

Given **enough labelled data**, we could have an object that is literally made up of 4 pixels, and as long as the classes are visually distinct enough, we will still be able to detect it.

![tetris blocks](../assets/images/hung-up-over-pixels/tetris.jpg)

## What then should be the question?

The question is no longer about how many minimum pixels we need, but whether our objects are still visually distinguishable at a certain range and resolution — if a human can look at the same frame & pick out the objects, then a machine can too with training on **enough labelled data**.

## Open Source Models and Datasets

If we really need to gather some form of initial sensing, I guess what we can get from open source models and datasets might be a good start? At the start, most projects will not have **labelled** deployment data and will rely on open-source pre-trained models, or open-source datasets that are a close proxy. If we are able to get the pixel size distributions of the wanted classes in the proxy datasets, then we might be able to get a good grasp on what is already feasible with current open-source models. Then again, once we get real deployment data and re-train, our performance is almost guaranteed to improve and once again that magic initial minimum pixel number is moot.  

## Conclusion

End of day, of course we could just pluck some logical numbers from out of the air and put it in the project document, but really, it's not meaningful to ask this. Get the **data** sorted out and annotate the **ground-truth labels**, then we can talk performance metrics in terms of accuracy, and how it trades off with target sizes under the deployment conditions.

## Epilogue

You might get the sense that we've emphasized on data quite a bit over here, well, because it is that important. The state of ML research might revolve around pushing that few 0.1% on academic benchmark datasets through proposing new models/training strategies, but really, in real world applications, a data-centric approach is what will bring you significant improvements to your model performance. But that's a story for another time ;)
