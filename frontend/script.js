const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const resultDiv = document.getElementById("result");
const confidenceDiv = document.getElementById("confidence");
const heatmapImg = document.getElementById("heatmapImg");
const heatmapBtn = document.querySelector(".heatmap-btn");

let intervalId = null;
let lastImageId = null;

// Open webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(() => alert("Camera access denied"));

function startDetection() {
    if (intervalId) return;

    heatmapBtn.disabled = true;
    heatmapImg.style.display = "none";

    intervalId = setInterval(() => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");

            fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                lastImageId = data.image_id;

                resultDiv.innerText = data.prediction.toUpperCase();
                confidenceDiv.innerText = `Confidence: ${data.confidence}%`;

                resultDiv.className =
                    data.prediction === "Fake" ? "status fake" : "status real";
            });
        }, "image/jpeg");

    }, 1000);
}

function stopDetection() {
    clearInterval(intervalId);
    intervalId = null;

    resultDiv.innerText = "Stopped";
    confidenceDiv.innerText = "";

    if (lastImageId) {
        heatmapBtn.disabled = false;
    }
}

function showHeatmap() {
    if (!lastImageId) {
        alert("Please start and stop detection first.");
        return;
    }

    fetch(`http://127.0.0.1:8000/heatmap-only/${lastImageId}`)
        .then(res => res.json())
        .then(data => {
            heatmapImg.src =
                "http://127.0.0.1:8000" + data.heatmap_url;
            heatmapImg.style.display = "block";
        })
        .catch(() => alert("Heatmap generation failed"));
}
