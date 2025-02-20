---
layout: default
title:  Home
---

<hr>

<center>

![Cuphead Cover Art](assets/cuphead_cover_art.jpeg)

</center>


**Summary**
Our project involves training a reinforcement learning agent to play the game *Cuphead*. We use a YOLO object detection model to read the contents of the screen, and trained an RL network to make decisions and key inputs based on what is seen by the model. Our goal is to beat one boss fight (root pack).


<center>

![Cuphead AI Image](assets/CUPHEAD_RL.png)

</center>



**Reports:**
 - [Proposal](proposal.html) 
 - [Status](status.html) 
 - [Final](final.html) 


**Source code:** 
https://github.com/simonxcao/bots

<hr>

**Sources**

- Our main inspiration and motivation to do this project was this YouTube videio by Develeper, which showed a functioning Cuphead agent. This video has some differences to our project such as their agent training on a very different boss, and their agent using tensorflow, while we used pytorch. Still, the video gave us inspiration of using an object detection model and training RL agent to output keypresses based on the locations of objects that the model sees. 
 https://www.youtube.com/watch?v=wipq--gdIGM
 ![Video Thumbnail](assets/video_screenshot.jpg)

 - This is another video that we found helpful. It talks about object detection models in the game *Grand Theft Auto*.
 https://www.youtube.com/watch?v=hNv854R1Guo

 - This page was helpful in creating the object detection model. 
 https://docs.ultralytics.com/models/yolo11/
 - We also created our own Cuphead dataset for object detection here:
 https://app.roboflow.com/cuphead-pj7jz/cuphead-objects/6
 












