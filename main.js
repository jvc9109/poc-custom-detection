import './style.css'
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

tf.setBackend('webgl');

const model = await loadGraphModel("https://raw.githubusercontent.com/jvc9109/models-tf-poc/main/web_model_accurate/model.json");


document.querySelector('#app').innerHTML = `
  <div>
    <div id="video-wrapper">
      <div id="overlay">
      </div>
      <img src="" />
      <canvas id="canvas" width="500" height="400"></canvas>
      <video id="video" autoplay></video>
    </div>
    <input type="range" min="0" max="20" value="0"/>
    <button>Take another picture</button>
  </div>
`;

const video = document.querySelector("video");
const button = document.querySelector("button");
const image = document.querySelector("img");
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");
const input = document.querySelector("input");

const classesDir = {
    1: {
        name: 'id_card',
        id: 1,
    },
    2: {
        name: 'Other',
        id: 2,
    }
}

const showPredictionBox = true;

let numHits = 0;
let interval;
let threshold = 0.9;

button.addEventListener('click', () => {
    image.classList.remove('is-visible');
    button.classList.remove('is-visible');
    numHits = 0;
    image.src = '';
    accessCamera();
});

const accessCamera = async () => {
    const stream = await navigator.mediaDevices
        .getUserMedia({
            audio: false,
            video: { width: 600, height: 500 },
        })
    video.srcObject = stream
};

function isInPostition(pos) {
    const [x, y] = pos;
    return  x >= 175 && x <= 205 || y <= 134 && y >= 165;
}

const processInput = (video_frame) => {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    return  tfimg.transpose([0,1,2]).expandDims();
};

const buildDetectedObjects = (scores, threshold, boxes, classes, classesDir) => {
    const detectionObjects = []
    var video_frame = document.getElementById('video');
    scores[0].forEach((score, i) => {
        if (score > threshold) {
            const bbox = [];
            const minY = boxes[0][i][0] * video_frame.offsetHeight;
            const minX = boxes[0][i][1] * video_frame.offsetWidth;
            const maxY = boxes[0][i][2] * video_frame.offsetHeight;
            const maxX = boxes[0][i][3] * video_frame.offsetWidth;
            bbox[0] = minX;
            bbox[1] = minY;
            bbox[2] = maxX - minX;
            bbox[3] = maxY - minY;
            detectionObjects.push({
                class: classes[i],
                label: classesDir[classes[i]].name,
                score: score.toFixed(4),
                bbox: bbox
            })
        }
    })
    return detectionObjects;
}

const detectFrame = async () => {
    tf.engine().startScope();
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const predictions = await model.executeAsync(processInput(video))

    input.value = numHits;
    if (numHits >= 20) {
        captureImage();
    }

    ctx.stroke();

    const boxes = predictions[7].arraySync();
    const scores = predictions[1].arraySync();
    const classes = predictions[4].dataSync();

    let detections = buildDetectedObjects(scores, threshold, boxes, classes, classesDir);

    detections.forEach(item => {
        const x = item['bbox'][0];
        const y = item['bbox'][1];
        const width = item['bbox'][2];
        const height = item['bbox'][3];

        // Draw the bounding box.
        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);

        // Draw the label background.
        ctx.fillStyle = "#00FFFF";
        const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
        const textHeight = parseInt(font, 10); // base 10
        ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    })

    detections.forEach(item => {
        const x = item['bbox'][0];
        const y = item['bbox'][1];

        // Draw the text last to ensure it's on top.
        ctx.fillStyle = "#000000";
        ctx.fillText(item["label"] + " " + (100*item["score"]).toFixed(2) + "%", x, y);
    });


    tf.engine().endScope();

};

video.addEventListener("loadeddata", async () => {
    interval = setInterval(detectFrame, 400);
});

function captureImage() {
    image.classList.add('is-visible');
    button.classList.add('is-visible');

    const canvas = document.createElement('canvas');
    canvas.width = 500;
    canvas.height = 400;
    canvas.getContext('2d').drawImage(video, 0, 0);

    image.src = canvas.toDataURL('image/png');

    clearTimeout(interval)
}

accessCamera();
