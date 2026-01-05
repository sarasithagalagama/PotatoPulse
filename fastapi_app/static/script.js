/* Elements */
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const previewContainer = document.getElementById("preview-container");
const previewImg = document.getElementById("preview-img");
const removeBtn = document.getElementById("remove-btn");
const analyzeBtn = document.getElementById("analyze-btn");

const resultCard = document.getElementById("result-card");
const diagnosisText = document.getElementById("diagnosis-text");
const confidenceText = document.getElementById("confidence-text");
const barsContainer = document.getElementById("bars-container");
const insightText = document.getElementById("insight-text");
const treatmentList = document.getElementById("treatment-list");

const accordionBtn = document.querySelector(".accordion-header");
const accordion = document.querySelector(".accordion");

const chatToggle = document.getElementById("chat-toggle");
const chatWindow = document.getElementById("chat-window");
const closeChat = document.getElementById("close-chat");
const themeToggle = document.querySelector(".theme-toggle");

// Theme Toggle Logic
if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    const isLight = document.body.classList.contains("light-mode");
    localStorage.setItem("theme", isLight ? "light" : "dark");
  });
}

// Load saved preference
if (localStorage.getItem("theme") === "light") {
  document.body.classList.add("light-mode");
}

let currentFile = null;

/* 1. File Handling */
dropZone.addEventListener("click", (e) => {
  if (e.target !== removeBtn) {
    fileInput.click();
  }
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.backgroundColor = "rgba(255,255,255,0.05)";
});

dropZone.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dropZone.style.backgroundColor = "transparent";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.backgroundColor = "transparent";
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

removeBtn.addEventListener("click", (e) => {
  e.stopPropagation(); // prevent triggering dropzone click
  clearFile();
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file.");
    return;
  }
  currentFile = file;

  // Show Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadPlaceholder.classList.add("hidden");
    previewContainer.classList.remove("hidden");

    // Enable Button
    analyzeBtn.disabled = false;
    analyzeBtn.classList.add("active");
  };
  reader.readAsDataURL(file);

  // Hide previous results if any
  resultCard.classList.add("hidden");
}

function clearFile() {
  currentFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewContainer.classList.add("hidden");
  uploadPlaceholder.classList.remove("hidden");
  analyzeBtn.disabled = true;
  analyzeBtn.classList.remove("active");
  resultCard.classList.add("hidden");
}

/* 2. Analysis */
analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  analyzeBtn.textContent = "Processing...";
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Analysis Failed");

    const data = await response.json();

    // Slight delay to simulate "Analysis Protocol"
    setTimeout(() => {
      showResults(data);
      analyzeBtn.textContent = "RUN ANALYSIS PROTOCOL";
      analyzeBtn.disabled = false;
    }, 800);
  } catch (err) {
    console.error(err);
    alert("Error during analysis.");
    analyzeBtn.textContent = "RUN ANALYSIS PROTOCOL";
    analyzeBtn.disabled = false;
  }
});

function showResults(data) {
  resultCard.classList.remove("hidden");

  // 1. Primary Diagnosis
  diagnosisText.textContent = data.class;
  confidenceText.textContent = `${data.confidence.toFixed(1)}% Confidence`;

  // 2. Bars (Confidence Distribution)
  barsContainer.innerHTML = "";
  // Sort distribution by value desc
  const sortedDist = Object.entries(data.distribution).sort(
    (a, b) => b[1] - a[1]
  );

  sortedDist.forEach(([label, score]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
            <div class="bar-label">${label}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width: 0%"></div>
            </div>
            <div class="bar-value">${score.toFixed(1)}%</div>
        `;
    barsContainer.appendChild(row);

    // Trigger animation
    setTimeout(() => {
      row.querySelector(".bar-fill").style.width = `${score}%`;
    }, 100);
  });

  // 3. Dynamic Insights & Roadmap
  if (data.class === "Healthy") {
    insightText.textContent =
      "Analysis indicates healthy cellular structures. Chlorophyll levels appear normal with no necrosis markers.";
    treatmentList.innerHTML = `
            <li>Continue standard irrigation schedule (1-2 inches/week).</li>
            <li>Maintain regular scouting every 3-5 days.</li>
            <li>No fungicide application required at this stage.</li>
        `;
  } else if (data.class === "Early Blight") {
    insightText.textContent =
      "Detected characteristic concentric ring lesions indicative of Alternaria solani. Infection likely due to alternating wet/dry conditions.";
    treatmentList.innerHTML = `
            <li>Prune infected lower leaves to reduce inoculum.</li>
            <li>Apply Copper-based fungicides or Chlorothalonil.</li>
            <li>Improve air circulation by reducing plant density.</li>
        `;
  } else if (data.class === "Late Blight") {
    insightText.textContent =
      "Critical Alert: Identified water-soaked lesions consistent with Phytophthora infestans. High risk of rapid spread.";
    treatmentList.innerHTML = `
            <li>Review humidity control immediately; reduce leaf wetness.</li>
            <li>Apply systemic fungicides (e.g., Metalaxyl/Mefenoxam).</li>
            <li>Destroy severely infected plants to prevent sporulation.</li>
        `;
  }

  // Scroll to result
  resultCard.scrollIntoView({ behavior: "smooth" });
}

/* 3. UI Interactions */
// Accordion
accordionBtn.addEventListener("click", () => {
  accordion.classList.toggle("open");
});

// Chat
chatToggle.addEventListener("click", () => {
  chatWindow.classList.toggle("hidden");
});

closeChat.addEventListener("click", () => {
  chatWindow.classList.add("hidden");
});

// Copy Report
document.getElementById("copy-report-btn").addEventListener("click", () => {
  const text = `PotatoPulse X1 Report\nDiagnosis: ${diagnosisText.textContent}\nConfidence: ${confidenceText.textContent}\nNotes: ${insightText.textContent}`;
  navigator.clipboard.writeText(text).then(() => {
    alert("Report copied to clipboard");
  });
});
