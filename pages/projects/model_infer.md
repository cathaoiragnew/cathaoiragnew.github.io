# YOLOv8 Object Detection in Markdown

This is an interactive object detection demo using **YOLOv8** with **ONNX**.

Upload an image, and the model will process it. The original image and the image with detected objects will be displayed.

## Upload an Image

<input id="uploadInput" type="file"/>
<div style="display: flex; gap: 20px;">
    <div>
        <h3>Original Image</h3>
        <canvas id="originalCanvas"></canvas>
    </div>
    <div>
        <h3>Processed Image</h3>
        <canvas id="processedCanvas"></canvas>
    </div>
</div>

<div style="width: 100%; margin-top: 10px;">
    <progress id="progressBar" value="0" max="100" style="width: 100%;"></progress>
</div>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

<script>
    const input = document.getElementById("uploadInput");
    input.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        showOriginalImage(file);
        document.getElementById("progressBar").value = 20;

        const boxes = await detectObjects(file);
        document.getElementById("progressBar").value = 80;

        drawImageAndBoxes(file, boxes);
        document.getElementById("progressBar").value = 100;
    });

    function showOriginalImage(file) {
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = () => {
            const canvas = document.getElementById("originalCanvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
        };
    }

    function drawImageAndBoxes(file, boxes) {
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = () => {
            const canvas = document.getElementById("processedCanvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.font = "18px serif";

            boxes.forEach(([x1, y1, x2, y2, label]) => {
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.fillStyle = "#00ff00";
                const width = ctx.measureText(label).width;
                ctx.fillRect(x1, y1, width + 10, 25);
                ctx.fillStyle = "#000000";
                ctx.fillText(label, x1, y1 + 18);
            });
        };
    }

    async function detectObjects(file) {
        const [input, img_width, img_height] = await prepareInput(file);
        const output = await runModel(input);
        return processOutput(output, img_width, img_height);
    }

    async function prepareInput(file) {
        return new Promise((resolve) => {
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                const [img_width, img_height] = [img.width, img.height];
                const canvas = document.createElement("canvas");
                canvas.width = 640;
                canvas.height = 640;
                const context = canvas.getContext("2d");
                context.drawImage(img, 0, 0, 640, 640);
                const imgData = context.getImageData(0, 0, 640, 640);
                const pixels = imgData.data;

                const red = [], green = [], blue = [];
                for (let i = 0; i < pixels.length; i += 4) {
                    red.push(pixels[i] / 255.0);
                    green.push(pixels[i + 1] / 255.0);
                    blue.push(pixels[i + 2] / 255.0);
                }
                const input = [...red, ...green, ...blue];
                resolve([input, img_width, img_height]);
            };
        });
    }

    async function runModel(input) {
        document.getElementById("progressBar").value = 50;
        const model = await ort.InferenceSession.create("yolov8m.onnx");
        input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);
        const outputs = await model.run({ images: input });
        return outputs["output0"].data;
    }

    function processOutput(output, img_width, img_height) {
        let boxes = [];
        for (let i = 0; i < 8400; i++) {
            const [class_id, prob] = [...Array(80).keys()]
                .map((col) => [col, output[8400 * (col + 4) + i]])
                .reduce((accum, item) => (item[1] > accum[1] ? item : accum), [0, 0]);

            if (prob < 0.5) continue;
            const label = yolo_classes[class_id];
            const xc = output[i], yc = output[8400 + i];
            const w = output[2 * 8400 + i], h = output[3 * 8400 + i];
            const x1 = (xc - w / 2) / 640 * img_width;
            const y1 = (yc - h / 2) / 640 * img_height;
            const x2 = (xc + w / 2) / 640 * img_width;
            const y2 = (yc + h / 2) / 640 * img_height;
            boxes.push([x1, y1, x2, y2, label, prob]);
        }

        boxes = boxes.sort((a, b) => b[5] - a[5]);
        const result = [];
        while (boxes.length > 0) {
            result.push(boxes[0]);
            boxes = boxes.filter((box) => iou(boxes[0], box) < 0.7);
        }
        return result;
    }

    function iou(box1, box2) {
        return intersection(box1, box2) / union(box1, box2);
    }

    function union(box1, box2) {
        const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        return area1 + area2 - intersection(box1, box2);
    }

    function intersection(box1, box2) {
        const x1 = Math.max(box1[0], box2[0]);
        const y1 = Math.max(box1[1], box2[1]);
        const x2 = Math.min(box1[2], box2[2]);
        const y2 = Math.min(box1[3], box2[3]);
        return (x2 - x1) * (y2 - y1);
    }

    const yolo_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'skateboard', 'surfboard', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake'
    ];
</script>
