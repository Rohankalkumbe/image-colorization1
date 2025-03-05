function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    let originalImage = document.getElementById("originalImage");
    let colorizedImage = document.getElementById("colorizedImage");
    let downloadBtn = document.getElementById("downloadBtn");

    if (!fileInput.files.length) {
        alert("Please select an image to upload.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    let reader = new FileReader();
    reader.onload = function (e) {
        originalImage.src = e.target.result;
        originalImage.style.display = "block";
    };
    reader.readAsDataURL(fileInput.files[0]);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        let imgUrl = URL.createObjectURL(blob);
        colorizedImage.src = imgUrl;
        colorizedImage.style.display = "block";
        downloadBtn.href = imgUrl;
        downloadBtn.style.display = "block";
    })
    .catch(error => console.error("Error:", error));
}
