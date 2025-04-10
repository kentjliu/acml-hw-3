document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('imageUpload');
    const preview = document.getElementById('preview');
    const classifyBtn = document.getElementById('classifyBtn');
    const result = document.getElementById('result');
    
    let selectedFile = null;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            selectedFile = files[0];
            previewFile(selectedFile);
            classifyBtn.disabled = false;
        }
    }

    function previewFile(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            const img = document.createElement('img');
            img.src = reader.result;
            preview.innerHTML = '';
            preview.appendChild(img);
        }
    }

    classifyBtn.addEventListener('click', function() {
        if (!selectedFile) {
            return;
        }

        // Show loading state
        result.innerHTML = '<p>Classifying image...</p>';
        classifyBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // Get current URL to build the API URL
        // This assumes the inference service is named "model-inference-service" in the same namespace
        // In a real app, you'd configure this more carefully
        // For local testing, you can change this to your inference service's external IP
        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                result.innerHTML = `<p>Error: ${data.error}</p>`;
            } else {
                const confidencePercent = (data.confidence * 100).toFixed(2);
                result.innerHTML = `
                    <div class="prediction">
                        <span class="prediction-label">Class:</span>
                        <span class="prediction-value">${data.class}</span>
                    </div>
                    <div class="prediction">
                        <span class="prediction-label">Confidence:</span>
                        <span class="prediction-value">${confidencePercent}%</span>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: ${confidencePercent}%"></div>
                    </div>
                `;
            }
        })
        .catch(error => {
            result.innerHTML = `<p>Error: ${error.message}</p>`;
        })
        .finally(() => {
            classifyBtn.disabled = false;
        });
    });
});
