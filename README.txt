ABSTRACT
There are thousands of species of plant & flowers & it’s very difficult for any individual to remember each & every flower with precise details however currently there is no any reliable application for students & researchers to identify any plant, anytime, anywhere with just one capture. 
The reason behind it that Deep Learning Algorithms required very specified, high quality & precisely labelled data. 
While there are still chances of error because of similarity in plants so we captured images with specific methods to gain high quality dataset. 
In this project, we going to initialize an app that will identify plant flowers with reliable prediction. 
First of all, we gathered data in specifics periods (day-light & evening) & Angles entire plant, flower frontal- and lateral view, flower top and backside view. 
The algorithm we are going to use in SSD Mobile Net V2. 
We gathered 10 classes of data in this initial project & in each class there are approximately 200 images according to a specific data-gathering technique. We design model so it can classify image with localization so that it can identify multiple flowers in a single image.
After training the deep learning model we converted it in TFlite model so that it become compatible to deploy in android app kotlin.
After that we Design Android App to take input images show results description about flowers & deploy TFlite converted model in Android App. 
Android App can identify 10 specific flowers & show results on real time.

