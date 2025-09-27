// Resume Upload Functionality
document.addEventListener('DOMContentLoaded', function() {
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');

  // Click to open file picker
  uploadArea.addEventListener('click', () => fileInput.click());

  // Drag and drop
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });

  uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
  });

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // Handle file upload
  function handleFile(file) {
    const allowedTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];

    // Validate type
    if (!allowedTypes.includes(file.type)) {
      showNotification('Please upload a PDF, DOC, or DOCX file.', 'error');
      return;
    }

    // Validate size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      showNotification('File size must be less than 10MB.', 'error');
      return;
    }

    // Show upload progress (fake for now)
    showUploadProgress(file.name);

    // ✅ At this point, the file is valid.
    // Later you will add your dataset/backend logic here.
    setTimeout(() => {
      showUploadSuccess(file.name);
    }, 2000);
  }

  // Show upload progress
  function showUploadProgress(fileName) {
    const uploadContent = document.querySelector('.upload-content');
    uploadContent.innerHTML = `
      <div class="upload-text">
        <h3>Uploading ${fileName}...</h3>
        <p>Please wait while we process your file.</p>
        <div class="progress-bar"><div class="progress-fill"></div></div>
      </div>
    `;

    const style = document.createElement('style');
    style.textContent = `
      .progress-bar {
        width: 200px;
        height: 4px;
        background: #ddd;
        border-radius: 2px;
        margin: 16px auto 0;
        overflow: hidden;
      }
      .progress-fill {
        height: 100%;
        background: linear-gradient(to right, #4facfe, #00f2fe);
        width: 0%;
        animation: fillProgress 2s ease-out forwards;
      }
      @keyframes fillProgress {
        to { width: 100%; }
      }
    `;
    document.head.appendChild(style);
  }

  // Show success state (no dataset yet)
  function showUploadSuccess(fileName) {
    const uploadContent = document.querySelector('.upload-content');
    uploadContent.innerHTML = `
      <div class="upload-text">
        <h3>Upload Complete!</h3>
        <p>Your file <strong>${fileName}</strong> has been uploaded successfully.</p>
      </div>
    `;
    showNotification("File uploaded successfully!", "success");
  }

  // Notifications
  function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    const style = document.createElement('style');
    style.textContent = `
      .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 6px;
        color: white;
        font-weight: 500;
        z-index: 1001;
        animation: slideInRight 0.3s ease, fadeOut 0.3s ease 2.7s forwards;
      }
      .notification.success { background: #28a745; }
      .notification.error { background: #dc3545; }
      .notification.info { background: #007bff; }
      @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
      @keyframes fadeOut {
        to { opacity: 0; transform: translateX(100%); }
      }
    `;
    document.head.appendChild(style);

    document.body.appendChild(notification);

    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
        style.remove();
      }
    }, 3000);
  }
});