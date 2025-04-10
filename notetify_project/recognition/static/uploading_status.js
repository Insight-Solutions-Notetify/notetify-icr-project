const form = document.getElementById('upload-form')
const imageInput = document.getElementById('image-input')
const results = document.getElementById('results')
const statusText = document.getElementById('status');
const recognizedText = document.getElementById('recognized-text');
const previewImage = document.getElementById('preview-image');
const processedImage = document.getElementById('processed-image');
const textBlock = document.getElementById('text-block');
const processedBlock = document.getElementById('processed-block');

form.addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = new FormData();
    const file = imageInput.files[0];
    formData.append('image', file);

    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    fetch('/api/upload/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrfToken,
        },
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            const imageUrl = URL.createObjectURL(file);
            console.log("Changing status");
            previewImage.src = imageUrl;
            results.classList.remove('hidden');
            statusText.innerText = "Processing...";

            pollStatus(data.id);
        })
        .catch(err => {
            console.error("Upload failed:", err);
            statusText.innerText = "Upload failed. Please try again.";
        });
});

function pollStatus(taskId, attempts = 0) {

    function poll() {
        fetch(`/api/status/${taskId}/`)
            .then(res => {
                if (!res.ok) {
                    if (res.status == 404) {
                        // Try again shortly - boject may not be created yet
                        setTimeout(() => pollStatus(taskId, attempts + 1), 1000);
                    } else {
                        statusText.innerHTML = "Error: Task not found.";
                    }
                    return;
                }
                return res.json();
            })
            .then(data => {
                if(!data) return;


                if (data.processed) {
                    statusText.innerText = "Processing Complete!";
                    recognizedText.innerHTML = data.text;
                    if (data.processed_url) {
                        textBlock.classList.remove('hidden');
                        processedBlock.classList.remove('hidden');
                        processedImage.src = data.processed_url
                        // document.getElementById('processed-link').href = data.processed_url
                    }
                }
                else {
                    statusText.innerHTML = "Processing...";
                    setTimeout(() => pollStatus(taskId, attempts + 1), 1000);
                }
            })
            .catch(err => {
                statusText.innerHTML = "Error checking status.";
                console.error(err);
            });
    }
    poll();
}