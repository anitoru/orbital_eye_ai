// Upload form: show selected file name, drag & drop
const input = document.getElementById("oct_image");
const fileName = document.getElementById("file-name");
const dropzone = document.getElementById("dropzone");

if (input && fileName) {
  input.addEventListener("change", () => {
    fileName.textContent = input.files.length ? input.files[0].name : "No file selected";
  });
}
if (dropzone && input) {
  ["dragenter", "dragover"].forEach(evt => dropzone.addEventListener(evt, e => {
    e.preventDefault(); dropzone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach(evt => dropzone.addEventListener(evt, e => {
    e.preventDefault(); dropzone.classList.remove("dragging");
  }));
  dropzone.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      fileName.textContent = input.files[0].name;
    }
  });
}

// Compare: enable the button only when exactly two examinations are ticked
const compareBtn = document.getElementById("compare-btn");
const boxes = document.querySelectorAll(".compare-box");
if (compareBtn && boxes.length) {
  const update = () => {
    const n = [...boxes].filter(b => b.checked).length;
    compareBtn.disabled = n !== 2;
    compareBtn.textContent = `Compare selected (${n}/2)`;
  };
  boxes.forEach(b => b.addEventListener("change", update));
}

// Shared Chart.js line chart in the ORBITAL EYE palette
const ORBITAL_COLORS = ["#4779a9", "#2aa198", "#d89a3d", "#8b5cf6"];
function orbitalLineChart(canvasId, labels, series, yTitle) {
  const el = document.getElementById(canvasId);
  if (!el || typeof Chart === "undefined") return;
  let c = 0;
  new Chart(el, {
    type: "line",
    data: {
      labels,
      datasets: series.map(s => {
        const color = s.dashed ? "#d07a6b" : ORBITAL_COLORS[c++ % ORBITAL_COLORS.length];
        return {
          label: s.label, data: s.data, borderColor: color, backgroundColor: color,
          borderWidth: s.dashed ? 1 : 2, borderDash: s.dashed ? [6, 4] : [],
          pointRadius: s.dashed ? 0 : 4, spanGaps: true, tension: 0.2,
        };
      }),
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { font: { family: "Manrope", size: 11 }, boxWidth: 12 } } },
      scales: {
        x: { ticks: { font: { size: 10 }, maxRotation: 0, autoSkip: true }, grid: { display: false } },
        y: { title: { display: true, text: yTitle, font: { size: 10 } }, grid: { color: "#edf1f4" } },
      },
    },
  });
}
