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
