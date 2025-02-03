---
layout: page
title: Return of the Doorbell
---

Revisting the [smart doorbell project](https://cathaoiragnew.github.io/pages/projects/doorbell/), I have now updated this project to provide more features. A brief overview is given for each of the steps.
For this project I do not fine-tune any model, this is the off-the-shelf performance for each of the models. 

### 1. Video Frame Processing with OpenCV
The system begins by reading each frame from video streams using OpenCV.

### 2. Get Depth Map Estimates
Using the [DINOv2](https://huggingface.co/facebook/dpt-dinov2-small-kitti) model from HuggingFace, depth maps (in metres) are generated for each frame. These maps estimate the distance of each pixel from the camera, aiding in understanding spatial relationships between objects. 

 <p style="text-align: center;">
   <img src="/assets/img/depth_ex.jpg" alt="Depth Map Example" style="max-width: 100%; height: auto;" />
 </p>  

### 3. Perform Object Detection
The [DETR](https://huggingface.co/facebook/detr-resnet-50) model is employed for object detection, identifying and locating objects within frames. Following detection, [DeepSORT](https://github.com/nwojke/deep_sort) is used to track these objects across frames, assigning unique identifiers to each. Using the tracked objects, speed estimates are calculated for each object by analyzing its movement across consecutive frames. This is done by computing the distance an object moves between frames, along with the frame rate of the video, to estimate the speed at which the object is traveling. These are not perfect and will compound errors from the depth maps, but the estimates may still provide some useful information. We could apply some moving averages to reduce the uncertainty in distance and speed estimates, or kalman filters if we had real measurements. However for the sake of this project I don't implement this. 

### 4. Pose estimation
For each detected person, [MediaPose](https://github.com/google-ai-edge/mediapipe) is used for pose estimation. This is done by cropping the region of interest based on the person's bounding box and passing it to MediaPose.

### 5. Face Detection & Recognition using 
The [InsightFace](https://github.com/deepinsight/insightface) library is used for face detection and embedding. A RetinaNet model fine-tuned for face detection identifies faces within frames, and embeddings are generated for recognition purposes.

#### 5.1 Face Labeling Interface
 A custom function is developed to read images, detect faces and open a GUI for labelling detected faces (quickly developed for functionality not aesthetics/UI). This facilitates the creation of a labelled dataset for face recognition.

 <p style="text-align: center;">
   <img src="/assets/img/face_embed_collect.PNG" alt="Face Embedding Labeller" style="max-width: 100%; height: auto;" />
 </p>  

#### 5.2 Vector Search for Embedding Comparison
  The [Faiss](https://github.com/facebookresearch/faiss) library is employed to perform vector searches. Normalized embeddings are compared using L2 distance to identify the closest matches between detected faces and labeled embeddings. 


### 6. Text Detection with EasyOCR
Finally [EasyOCR](https://github.com/JaidedAI/EasyOCR) is used to detect and extract text from images. This capability enhances the system's ability to interpret and respond to textual information present in the environment.

Here is a quick image to show easyOCR and Facial Reconition in action. Blurring is used to protect identities. We can see the red outline for the detected face refers to an unknown face recognised. It will be green if face is recognised (see example video below were it detects and recognises me). 

 <p style="text-align: center;">
   <img src="/assets/img/frame_unknown_1_trim_output_10.jpg" alt="Depth Map Example" style="max-width: 100%; height: auto;" />
 </p>  

### 7. Example Video

<center>
<!-- Video Container (Responsive) -->
<div class="responsive-video">
  <iframe src="/assets/img/example_walking.mp4" type="video/mp4" 
          title="YouTube video player" 
          frameborder="0" 
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
          referrerpolicy="strict-origin-when-cross-origin" 
          allowfullscreen></iframe>
</div>
</center>


### Conclusion
The enhanced smart doorbell system now offers object detection, tracking, pose estimation, face recognition, text interpretation, distance and speed estimates. This information can be used to build high level based rule systems such as fall detection, detection of persons in eclusion zones etc. 

**Note:** I tried to explicitly detect license plates using available models on HuggingFace, to allow for vechicle recoginition, but found no open-source models performed well for my given videos. This may be due to several reasons, video quality, angle/distance to license plate, possibly lack of Irish licenses in training data etc. A workaround could involve high-level filtering rules, such as using regex to extract specific license plate patterns or maintaining a dictionary of known plates. However, I chose not to pursue this, as the primary goal of this project is to provide the information needed for further rule-based automation. Lastly, I could fine-tune a model for my use-case, however I did not understake this as the purpose of this project is just to provide all the necessary information to build high level rules with off-the-shelf models. 



