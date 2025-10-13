document.addEventListener('DOMContentLoaded', function() {
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');
  const resultContainer = document.getElementById('resultContainer');
  const predictedJob = document.getElementById('predictedJob');
  const suitabilityScore = document.getElementById('suitabilityScore');
  const feedback = document.getElementById('feedback');
  const matchSkills = document.getElementById('matchSkills');
  const missingSkills = document.getElementById('missingSkills');
  const notificationContainer = document.getElementById('notificationContainer');

  // Click upload area to open file picker
  uploadArea.addEventListener('click', () => fileInput.click());

  // Drag & Drop
  uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
  });
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });

  // File input change
  fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  // Handle uploaded file
  function handleFile(file) {
    const allowedTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];

    if (!allowedTypes.includes(file.type)) {
      showNotification('Please upload a PDF, DOC, or DOCX file.', 'error');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      showNotification('File size must be less than 10MB.', 'error');
      return;
    }

    showNotification(`Analyzing ${file.name}...`, 'info');

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/upload', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      if(data.error) {
        showNotification(data.error, 'error');
        resultContainer.style.display = 'none';
      } else {
        showNotification('Analysis complete!', 'success');
        displayResult(data);
      }
    })
    .catch(err => {
      showNotification('Failed to upload file. Please try again.', 'error');
      resultContainer.style.display = 'none';
      console.error(err);
    });
  }

  // Display analysis result
  function displayResult(data) {
    predictedJob.textContent = data.predicted_job || 'N/A';
    suitabilityScore.textContent = data.suitability_score ? `${data.suitability_score}%` : 'N/A';
    feedback.textContent = data.feedback || 'No feedback available';
    matchSkills.textContent = data.match_skills ? data.match_skills.join(', ') : 'None';
    missingSkills.textContent = data.missing_skills ? data.missing_skills.join(', ') : 'None';
    resultContainer.style.display = 'block';
  }

  // Show notifications
  function showNotification(message, type='info') {
    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.textContent = message;
    notificationContainer.appendChild(notif);

    setTimeout(() => {
      notif.remove();
    }, 3000);
  }
});

