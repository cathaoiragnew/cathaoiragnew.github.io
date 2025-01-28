---
layout: page
title: Mask R-CNN Image Segmentation
---

# Mask R-CNN Object Segmentation

This page demonstrates the use of the Mask R-CNN model for object segmentation on uploaded images.

---

<div>
  <h3>Upload an Image for Segmentation</h3>
  <input type="file" id="fileInput" onchange="loadAndSegmentImage()"/>
  <br />
  <h4>Input Image:</h4>
  <img id="inputImage" width="300" alt="Uploaded image"/>
  <br />
  <h4>Segmentation Result:</h4>
  <img id="segmentationResult" width="300" alt="Segmentation result"/>
</div>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.10.0/dist/ort.min.js"></script>

<script>
  // URL for the Mask R-CNN model (ensure to replace with the direct URL)
  const modelURL = "https://github.com/onnx/models/releases/download/Computer_Vision/maskrcnn_resnet50_fpn_v2_Opset16.onnx";

  // Function to handle image upload and prediction
  async function loadAndSegmentImage() {
    const inputFile = document.getElementById('fileInput').files[0];
    if (!inputFile) {
      alert('Please upload an image first.');
      return;
    }

    // Load the image
    const img = document.getElementById('inputImage');
    img.src = URL.createObjectURL(inputFile);

    // Load the model
    try {
      const session = await ort.InferenceSession.create(modelURL); // Correct method call using ort
      console.log("Model loaded successfully!");

      // Prepare image for inference
      const imageTensor = await prepareImageForInference(inputFile);

      // Run the model to get predictions
      const results = await session.run({ images: imageTensor });

      // Post-process the segmentation results
      const segmentedImage = processSegmentationResults(results);

      // Display the segmented image
      const segmentationImageElement = document.getElementById('segmentationResult');
      segmentationImageElement.src = segmentedImage;
    } catch (error) {
      console.error("Error loading model:", error);
    }
  }

  // Helper function to convert image to tensor
  async function prepareImageForInference(imageFile) {
    const img = await loadImage(imageFile);
    
    // Resize and normalize the image (using Mask R-CNN input size: 800x800)
    const tensor = preprocessImage(img, 800, 800);
    
    // Returning a tensor that is compatible with the model
    return new ort.Tensor(tensor, 'float32');
  }

  // Load image into an HTMLImageElement
  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  // Preprocess the image (resize and normalize it to fit the model input)
  function preprocessImage(img, width, height) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    const normalizedData = new Float32Array(3 * width * height);

    // Normalize image data to the range [0, 1]
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255.0;
      const g = data[i + 1] / 255.0;
      const b = data[i + 2] / 255.0;
      const idx = (i / 4) * 3;
      normalizedData[idx] = r;
      normalizedData[idx + 1] = g;
      normalizedData[idx + 2] = b;
    }

    return normalizedData;
  }

  // Post-process segmentation results (to display masks)
  function processSegmentationResults(results) {
    // Assuming the model output includes boxes, masks, and labels
    const boxes = results[0].data;  // Bounding boxes
    const masks = results[1].data;  // Masks
    const labels = results[2].data; // Class labels

    // Create a canvas to draw the results
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const width = 800;  // Expected width for Mask R-CNN
    const height = 800; // Expected height for Mask R-CNN
    canvas.width = width;
    canvas.height = height;

    // Draw the masks
    for (let i = 0; i < masks.length; i++) {
      const mask = masks[i];
      const box = boxes[i];
      const x1 = box[0] * width;
      const y1 = box[1] * height;
      const x2 = box[2] * width;
      const y2 = box[3] * height;

      ctx.beginPath();
      ctx.rect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'red';
      ctx.stroke();

      // Draw the mask
      ctx.putImageData(mask, x1, y1);
    }

    return canvas.toDataURL();
  }
</script>
