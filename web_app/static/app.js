const fileInput = document.querySelector("#fileInput");
const dropzone = document.querySelector("#dropzone");
const predictButton = document.querySelector("#predictButton");
const statusText = document.querySelector("#statusText");
const scoreText = document.querySelector("#scoreText");
const labelText = document.querySelector("#labelText");
const fileName = document.querySelector("#fileName");

const originalImage = document.querySelector("#originalImage");
const overlayImage = document.querySelector("#overlayImage");
const heatmapImage = document.querySelector("#heatmapImage");
const maskImage = document.querySelector("#maskImage");

let selectedFile = null;

function setStatus(text) {
  statusText.textContent = text;
}

function setSelectedFile(file) {
  selectedFile = file;
  fileName.textContent = file ? file.name : "尚未選擇檔案";
  predictButton.disabled = !file;
  scoreText.textContent = "-";
  labelText.textContent = "-";

  if (file) {
    originalImage.src = URL.createObjectURL(file);
    setStatus("已選擇圖片");
  }
}

fileInput.addEventListener("change", (event) => {
  setSelectedFile(event.target.files[0]);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("isDragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("isDragging");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("isDragging");
  const file = event.dataTransfer.files[0];
  if (file) {
    fileInput.files = event.dataTransfer.files;
    setSelectedFile(file);
  }
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);

  predictButton.disabled = true;
  setStatus("模型辨識中");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "辨識失敗");
    }

    originalImage.src = payload.images.original;
    overlayImage.src = payload.images.overlay;
    heatmapImage.src = payload.images.heatmap;
    maskImage.src = payload.images.mask;
    scoreText.textContent = payload.score.toFixed(4);
    labelText.textContent = payload.label === "abnormal" ? "異常" : "正常";
    setStatus("完成");
  } catch (error) {
    setStatus(error.message);
  } finally {
    predictButton.disabled = false;
  }
});
