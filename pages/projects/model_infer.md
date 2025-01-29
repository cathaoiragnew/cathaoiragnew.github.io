---
layout: page
title: Model Inference
---

# YOLOv8 Object Detection in Markdown

This is an interactive object detection demo using **YOLOv8** with **ONNX**.

Upload an image, and the model will process it. The original image and the image with detected objects will be displayed.

## Upload an Image

<head>
    <meta charset="UTF-8">
    <title>YOLOv8 Object Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <style>
      canvas {
          display:block;
          border: 1px solid black;
          margin-top:10px;
      }
      #progress-container {
          width: 100%;
          background-color: #f3f3f3;
          margin-top: 10px;
      }
      #progress-bar {
          width: 0%;
          height: 30px;
          background-color: #4caf50;
          text-align: center;
          line-height: 30px;
          color: white;
      }
      .image-container {
          display: flex;
          justify-content: space-between;
      }
    </style>
</head>
<body>
    <input id="uploadInput" type="file"/>
    <div id="progress-container">
      <div id="progress-bar">0%</div>
    </div>
    <div class="image-container">
        <canvas id="originalImage"></canvas>
        <canvas id="resultImage"></canvas>
    </div>
    <script>
      /**
       * "Upload" button onClick handler: uploads selected image file
       * to backend, receives array of detected objects
       * and draws them on top of image
       */
       const input = document.getElementById("uploadInput");
       input.addEventListener("change",async(event) => {
           const boxes = await detect_objects_on_image(event.target.files[0]);
           draw_image_and_boxes(event.target.files[0],boxes);
       })

      /**
       * Function draws the image from provided file
       * and bounding boxes of detected objects on
       * top of the image
       * @param file Uploaded file object
       * @param boxes Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],...]
       */
      function draw_image_and_boxes(file,boxes) {
          const img = new Image()
          img.src = URL.createObjectURL(file);
          img.onload = () => {
              const originalCanvas = document.getElementById("originalImage");
              const resultCanvas = document.getElementById("resultImage");
              originalCanvas.width = img.width;
              originalCanvas.height = img.height;
              resultCanvas.width = img.width;
              resultCanvas.height = img.height;
              const originalCtx = originalCanvas.getContext("2d");
              const resultCtx = resultCanvas.getContext("2d");
              originalCtx.drawImage(img,0,0);
              resultCtx.drawImage(img,0,0);
              resultCtx.strokeStyle = "#00FF00";
              resultCtx.lineWidth = 3;
              resultCtx.font = "18px serif";
              boxes.forEach(([x1,y1,x2,y2,label]) => {
                  resultCtx.strokeRect(x1,y1,x2-x1,y2-y1);
                  resultCtx.fillStyle = "#00ff00";
                  const width = resultCtx.measureText(label).width;
                  resultCtx.fillRect(x1,y1,width+10,25);
                  resultCtx.fillStyle = "#000000";
                  resultCtx.fillText(label, x1, y1+18);
              });
          }
      }

      /**
       * Function receives an image, passes it through YOLOv8 neural network
       * and returns an array of detected objects and their bounding boxes
       * @param buf Input image body
       * @returns Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
       */
      async function detect_objects_on_image(buf) {
          updateProgressBar(20);
          const [input,img_width,img_height] = await prepare_input(buf);
          updateProgressBar(50);
          const output = await run_model(input);
          updateProgressBar(80);
          const boxes = process_output(output,img_width,img_height);
          updateProgressBar(100);
          return boxes;
      }

      /**
       * Function used to convert input image to tensor,
       * required as an input to YOLOv8 object detection
       * network.
       * @param buf Content of uploaded file
       * @returns Array of pixels
       */
      async function prepare_input(buf) {
          return new Promise(resolve => {
              const img = new Image();
              img.src = URL.createObjectURL(buf);
              img.onload = () => {
                  const [img_width,img_height] = [img.width, img.height]
                  const canvas = document.createElement("canvas");
                  canvas.width = 640;
                  canvas.height = 640;
                  const context = canvas.getContext("2d");
                  context.drawImage(img,0,0,640,640);
                  const imgData = context.getImageData(0,0,640,640);
                  const pixels = imgData.data;

                  const red = [], green = [], blue = [];
                  for (let index=0; index<pixels.length; index+=4) {
                      red.push(pixels[index]/255.0);
                      green.push(pixels[index+1]/255.0);
                      blue.push(pixels[index+2]/255.0);
                  }
                  const input = [...red, ...green, ...blue];
                  resolve([input, img_width, img_height]);
              }
          })
      }

      /**
       * Function used to pass provided input tensor to YOLOv8 neural network and return result
       * @param input Input pixels array
       * @returns Raw output of neural network as a flat array of numbers
       */
      async function run_model(input) {
          const model = await ort.InferenceSession.create("../yolov8m.onnx");
          input = new ort.Tensor(Float32Array.from(input),[1, 3, 640, 640]);
          const outputs = await model.run({images:input});
          return outputs["output0"].data;
      }

      /**
       * Function used to convert RAW output from YOLOv8 to an array of detected objects.
       * Each object contain the bounding box of this object, the type of object and the probability
       * @param output Raw output of YOLOv8 network
       * @param img_width Width of original image
       * @param img_height Height of original image
       * @returns Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
       */
      function process_output(output, img_width, img_height) {
          let boxes = [];
          for (let index=0;index<8400;index++) {
              const [class_id,prob] = [...Array(80).keys()]
                  .map(col => [col, output[8400*(col+4)+index]])
                  .reduce((accum, item) => item[1]>accum[1] ? item : accum,[0,0]);
              if (prob < 0.5) {
                  continue;
              }
              const label = yolo_classes[class_id];
              const xc = output[index];
              const yc = output[8400+index];
              const w = output[2*8400+index];
              const h = output[3*8400+index];
              const x1 = (xc-w/2)/640*img_width;
              const y1 = (yc-h/2)/640*img_height;
              const x2 = (xc+w/2)/640*img_width;
              const y2 = (yc+h/2)/640*img_height;
              boxes.push([x1,y1,x2,y2,label,prob]);
          }

          boxes = boxes.sort((box1,box2) => box2[5]-box1[5])
          const result = [];
          while (boxes.length>0) {
              result.push(boxes[0]);
              boxes = boxes.filter(box => iou(boxes[0],box)<0.7);
          }
          return result;
      }

      /**
       * Function calculates "Intersection-over-union" coefficient for specified two boxes
       * https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
       * @param box1 First box in format: [x1,y1,x2,y2,object_class,probability]
       * @param box2 Second box in format: [x1,y1,x2,y2,object_class,probability]
       * @returns Intersection over union ratio as a float number
       */
      function iou(box1,box2) {
          return intersection(box1,box2)/union(box1,box2);
      }

      /**
       * Function calculates union area of two boxes.
       *     :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
       *     :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
       *     :return: Area of the boxes union as a float number
       * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
       * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
       * @returns Area of the boxes union as a float number
       */
      function union(box1,box2) {
          const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
          const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
          const box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
          const box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
          return box1_area + box2_area - intersection(box1,box2)
      }

      /**
       * Function calculates intersection area of two boxes
       * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
       * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
       * @returns Area of intersection of the boxes as a float number
       */
      function intersection(box1,box2) {
          const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
          const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
          const x1 = Math.max(box1_x1,box2_x1);
          const y1 = Math.max(box1_y1,box2_y1);
          const x2 = Math.min(box1_x2,box2_x2);
          const y2 = Math.min(box1_y2,box2_y2);
          return (x2-x1)*(y2-y1)
      }

      /**
       * Array of YOLOv8 class labels
       */
      const yolo_classes = [
          'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
          'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
          'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
          'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
          'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
          'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
      ];

      /**
       * Function to update the progress bar
       * @param {number} percentage The percentage to set the progress bar to
       */
      function updateProgressBar(percentage) {
          const progressBar = document.getElementById("progress-bar");
          progressBar.style.width = percentage + "%";
          progressBar.innerHTML = percentage + "%";
      }
    </script>
</body>
