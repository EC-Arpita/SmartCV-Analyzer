document.addEventListener('DOMContentLoaded', function() {
  // --- DOM Element Selectors ---
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');
  const resultContainer = document.getElementById('resultContainer');
  const notificationContainer = document.getElementById('notificationContainer');

  const predictedJob = document.getElementById('predictedJob');
  const suitabilityScore = document.getElementById('suitabilityScore');
  const overallFeedbackSummary = document.getElementById('overallFeedbackSummary');
  const matchSkills = document.getElementById('matchSkills');
  const missingSkills = document.getElementById('missingSkills');
  const topPredictionsList = document.getElementById('topPredictionsList');
  const personalizedFeedbackList = document.getElementById('personalizedFeedbackList');
  const defaultFeedbackList = document.getElementById('defaultFeedbackList');
  const skillsBarChart = document.getElementById('skillsBarChart');
  const jobsPieChart = document.getElementById('jobsPieChart');

  // --- Event Listeners ---
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

  // --- Handle uploaded file ---
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
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        if (data.error) {
          showNotification(data.error, 'error');
          resultContainer.style.display = 'none';
        } else {
          showNotification('Analysis complete!', 'success');
          displayResult(data);
        }
      })
      .catch(err => {
        showNotification('Failed to connect to the analyzer server or parse response.', 'error');
        resultContainer.style.display = 'none';
        console.error("Fetch error:", err);
      });
  }

  // --- Helper Functions for Display ---
  function populateList(element, items, isPrediction = false) {
    element.innerHTML = '';
    if (items && items.length > 0) {
      items.forEach((item, index) => {
        const li = document.createElement('li');
        if (isPrediction) {
          li.innerHTML = `<strong>${index + 1}. ${item.job}</strong> <span>(${item.score}%)</span>`;
        } else {
          li.textContent = item;
        }
        element.appendChild(li);
      });
    } else {
      element.innerHTML = isPrediction
        ? '<li>No secondary job predictions available.</li>'
        : '<li>No feedback points provided.</li>';
    }
  }

  function displayGraphs(graphData) {
    if (graphData) {
      if (graphData.skills_bar_chart) {
        skillsBarChart.src = `data:image/png;base64,${graphData.skills_bar_chart}`;
      }
      if (graphData.jobs_pie_chart) {
        jobsPieChart.src = `data:image/png;base64,${graphData.jobs_pie_chart}`;
      }
    }
  }

  // --- Display analysis result ---
  function displayResult(data) {
    const topJob = data.predicted_jobs && data.predicted_jobs.length > 0
      ? data.predicted_jobs[0].job
      : 'N/A';
    predictedJob.textContent = topJob;
    suitabilityScore.textContent = data.suitability_score ? `${data.suitability_score}%` : 'N/A';
    overallFeedbackSummary.textContent = data.overall_feedback || 'No summary feedback available.';

    populateList(topPredictionsList, data.predicted_jobs, true);

    matchSkills.textContent = data.match_skills && data.match_skills.length > 0
      ? data.match_skills.join(', ')
      : 'None Found';
    missingSkills.textContent = data.missing_skills && data.missing_skills.length > 0
      ? data.missing_skills.join(', ')
      : 'None Missing';

    populateList(personalizedFeedbackList, data.personalized_feedback);
    populateList(defaultFeedbackList, data.default_feedback);

    displayGraphs(data.career_analytics_dashboard_data);

    resultContainer.style.display = 'grid';
    resultContainer.scrollIntoView({ behavior: 'smooth' });
  }

  // --- Show notifications ---
  function showNotification(message, type = 'info') {
    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.textContent = message;
    notificationContainer.appendChild(notif);

    setTimeout(() => {
      notif.remove();
    }, 3000);
  }
});

