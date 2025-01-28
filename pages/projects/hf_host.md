---
layout: page
title: HF hosted model 
---

Welcome to my computer vision demo! Below, you can upload an image and see the predictions overlaid by the DETR model.


## Upload Image

<div>
  <input type="file" id="fileInput" />
  <button onclick="loadAndSegmentImage()">Process Image</button>
</div>

<div>
  <h3>Original Image</h3>
  <img id="inputImage" style="max-width: 100%; height: auto;" />
</div>

<div>
  <h3>Predicted Segmentation</h3>
  <img id="segmentationResult" style="max-width: 100%; height: auto;" />
</div>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/onnxruntime-web.min.js"></script>


<script>
  // URL for the ONNX model hosted on GitHub Releases or another CORS-enabled server
  const modelURL = "https://github.com/cathaoiragnew/cathaoiragnew.github.io/releases/download/Model/detr.onnx";

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

    // Load the model (use a suitable JS library to handle ONNX models)
    const session = await ort.InferenceSession.create(modelURL);
    console.log("Model loaded successfully!");

    // Prepare image for inference (you can use a library to convert the image to tensor)
    const imageTensor = await prepareImageForInference(inputFile);

    // Run the model to get predictions
    const results = await session.run([imageTensor]);

    // Post-process results to show segmentation masks
    const segmentedImage = processSegmentationResults(results);

    // Show the segmented image
    const segmentationImageElement = document.getElementById('segmentationResult');
    segmentationImageElement.src = segmentedImage;
  }

  // Helper function to convert image to tensor
  async function prepareImageForInference(imageFile) {
    // This part involves converting the uploaded image to a format suitable for the ONNX model
    // You may need a helper function here, depending on your model's input format (e.g., resizing, normalization)
    // Returning a dummy tensor for now
    return new ort.Tensor('float32', new Float32Array(3 * 800 * 1066), [1, 3, 800, 1066]);
  }

  // Post-process segmentation results to generate an image URL
  function processSegmentationResults(results) {
    // This is where you would process the model output and convert it into an image format
    // For simplicity, let's assume this function returns a base64-encoded image string
    return 'data:image/png;base64,...'; // Replace with actual processing logic
  }
</script>
