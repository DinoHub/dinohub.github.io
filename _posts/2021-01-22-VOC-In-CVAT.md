---
layout: post
title:  "What we learnt from working with PASCAL VOC format in CVAT"
author: levan
categories: [CV]
tags: [Object Detection, Annotation]
image: assets/images/voc-in-cvat/cvat-upload.jpg
description: "What we learnt from working with PASCAL VOC format in CVAT"
featured: false
hidden: false
---

# What we learnt from working with PASCAL VOC format in CVAT

## Preface

In our journey to train/harvest a good person detector (look for us to learn more), we had to collate a multiple datasets into the same format (our favourite COCO format). One approach we took to do this was to load a dataset into CVAT and dump the annotations out in COCO format. It allowed us to look through the annotations for sanity check/modifications and also provided "free conversion".

Here I document the lessons I learnt from this, trivial it may be, but with the hope that the next person do not need to go through what I went through.

## The Dataset

The KAIST Multispectral Pedestrian Dataset provides annotations in their own format, but also provided some xml files in PASCAL VOC format.

Or so I thought. The XML files were not in exactly the right format to be uploaded into CVAT. Investigation involves dumping out in CVAT's Pascal VOC format to see how their XML files looks like.

Here is a good [link](https://github.com/openvinotoolkit/cvat/tree/develop/cvat/apps/dataset_manager/formats#pascal-voc-import) to CVAT's documentation on what it expects in Pascal VOC annotation imports, as well as other formats it accepts and dumps out in.

## Path Issues

Firstly, the filename value in the XMLs need to excatly match the filenames of the images uploaded to the CVAT task. If I uploaded `set08_V000_I00859.jpg`, then in my `.xml` file it needs to reflect the same filename:

```xml
<annotation>
...
  <filename>set08_V000_I00859.jpg</filename>
...
</annotation>

```

## Object Information Format

The dataset I was working with gave the bounding box information as follows:

```xml
...
<object>
    <name>person</name>
    <bndbox>
        <x>526</x>
        <y>240</y>
        <w>71</w>
        <h>184</h>
    </bndbox>
    <pose>unknown</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <occlusion>0</occlusion>
</object>
...
```

After investigation, what CVAT actually expects (else it will throw an error during importing of annotations):

```xml
...
<object>
    <name>person</name>
    <bndbox>
        <xmin>526</xmin>
        <xmax>597</xmax>
        <ymin>240</ymin>
        <ymax>424</ymax>
    </bndbox>
    <pose>unknown</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <occluded>0</occluded>
</object>
...
```
So yes, the following things to take note:

1. Bounding Box: Needs to be `xmin`, `xmax`, etc. instead of `x`,`y`,`w`,`h`.
2. Occlusion Tag: Can be passed on as well, but needs to be exactly the right key `occluded`, instead of `occlusion`.


That's all folks, thank you for coming to my ted talk.
