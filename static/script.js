/**
 * NeuroScan — Frontend Logic
 *
 * Flow:
 *  1. User drags/clicks → file selected → preview shown
 *  2. "Analyze Scan" → POST /predict with FormData → loading state
 *  3. Response → results section with animated ring + bar + images
 *  4. "Analyze Another Scan" → full reset
 */

'use strict';

/* ──────────────────────────────────────────────────────────
   DOM References
   ────────────────────────────────────────────────────────── */
const uploadSection  = document.getElementById('upload-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');

const dropzone       = document.getElementById('dropzone');
const fileInput      = document.getElementById('file-input');
const dzIdle         = document.getElementById('dz-idle');
const dzPreview      = document.getElementById('dz-preview');
const previewImg     = document.getElementById('preview-img');
const previewFilename= document.getElementById('preview-filename');
const analyzeBtn     = document.getElementById('analyze-btn');
const changeFileBtn  = document.getElementById('change-file-btn');

const stage1El       = document.getElementById('stage-1');
const stage2El       = document.getElementById('stage-2');
const stage3El       = document.getElementById('stage-3');

const dxClass        = document.getElementById('dx-class');
const dxDescription  = document.getElementById('dx-description');
const resultOriginal = document.getElementById('result-original');
const resultHeatmap  = document.getElementById('result-heatmap');
const resultLime     = document.getElementById('result-lime');
const newScanBtn     = document.getElementById('new-scan-btn');

const errorToast     = document.getElementById('error-toast');
const errorMsg       = document.getElementById('error-msg');
const closeToastBtn  = document.getElementById('close-toast');

/* ──────────────────────────────────────────────────────────
   Constants
   ────────────────────────────────────────────────────────── */

/** Tumor class display config */
const CLASS_CONFIG = {
  glioma: {
    label:       'Glioma',
    color:       '#5B5BD6',
    description: 'A tumour arising from glial cells in the brain or spine. '
               + 'Gliomas vary in aggressiveness. Further evaluation by a neurologist is recommended.',
  },
  meningioma: {
    label:       'Meningioma',
    color:       '#C47F00',
    description: 'A tumour arising from the meninges — the membranes surrounding the brain and spinal cord. '
               + 'Most are benign and slow-growing, though monitoring is important.',
  },
  pituitary: {
    label:       'Pituitary Tumor',
    color:       '#7C3AED',
    description: 'A tumour developing in the pituitary gland at the base of the brain. '
               + 'Many pituitary tumours are non-cancerous and respond well to treatment.',
  },
  notumor: {
    label:       'No Tumor Detected',
    color:       '#059669',
    description: 'The AI model did not identify characteristics associated with glioma, meningioma, '
               + 'or pituitary tumours in this scan. A clear result does not substitute clinical evaluation.',
  },
};

/* ──────────────────────────────────────────────────────────
   State
   ────────────────────────────────────────────────────────── */
let selectedFile = null;
let objectURL    = null;
let stageTimers  = [];
let toastTimer   = null;

/* ──────────────────────────────────────────────────────────
   Dropzone — Drag & Drop
   ────────────────────────────────────────────────────────── */
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', (e) => {
  // Only remove class if we actually left the dropzone (not a child element)
  if (!dropzone.contains(e.relatedTarget)) {
    dropzone.classList.remove('drag-over');
  }
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');

  const file = e.dataTransfer?.files[0];
  if (!file) return;

  if (!file.type.startsWith('image/')) {
    showError('Please upload an image file (JPEG, PNG, BMP, TIFF, or WebP).');
    return;
  }

  selectFile(file);
});

/* ──────────────────────────────────────────────────────────
   Dropzone — Click to Browse
   ────────────────────────────────────────────────────────── */
dropzone.addEventListener('click', (e) => {
  // Don't trigger file browse when clicking action buttons in preview
  if (e.target.closest('.btn') || e.target.closest('#change-file-btn')) return;
  // Only open browser if no file is selected (idle state) or from the idle area
  if (!selectedFile) {
    fileInput.click();
  }
});

dropzone.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && !selectedFile) {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) selectFile(file);
});

/* ──────────────────────────────────────────────────────────
   File Selection
   ────────────────────────────────────────────────────────── */
function selectFile(file) {
  // Revoke old object URL to free memory
  if (objectURL) {
    URL.revokeObjectURL(objectURL);
  }

  selectedFile = file;
  objectURL    = URL.createObjectURL(file);

  previewImg.src            = objectURL;
  previewFilename.textContent = file.name;

  dzIdle.classList.add('hidden');
  dzPreview.classList.remove('hidden');
}

/** "Change file" — go back to idle state */
changeFileBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});

function clearFile() {
  if (objectURL) {
    URL.revokeObjectURL(objectURL);
    objectURL = null;
  }
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewFilename.textContent = '—';
  dzPreview.classList.add('hidden');
  dzIdle.classList.remove('hidden');
}

/* ──────────────────────────────────────────────────────────
   Analyze Button
   ────────────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!selectedFile) return;
  runAnalysis();
});

async function runAnalysis() {
  enterLoadingState();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body:   formData,
    });

    if (!res.ok) {
      let detail = `Server error (${res.status})`;
      try {
        const body = await res.json();
        if (body?.detail) detail = body.detail;
      } catch { /* ignore parse errors */ }
      throw new Error(detail);
    }

    const data = await res.json();

    // Complete all stages visually before showing results
    clearStageTimers();
    setAllStagesDone();

    setTimeout(() => showResults(data), 500);

  } catch (err) {
    clearStageTimers();
    exitLoadingState();
    showError(err.message || 'Analysis failed. Please try again.');
  }
}

/* ──────────────────────────────────────────────────────────
   Loading State
   ────────────────────────────────────────────────────────── */
function enterLoadingState() {
  uploadSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  resultsSection.classList.remove('animate-in');
  loadingSection.classList.remove('hidden');

  // Reset all stages
  [stage1El, stage2El, stage3El].forEach(s => {
    s.classList.remove('is-active', 'is-done');
  });

  // Advance through stages with timed delays
  activateStage(stage1El);

  const t1 = setTimeout(() => {
    doneStage(stage1El);
    activateStage(stage2El);
  }, 1300);

  const t2 = setTimeout(() => {
    doneStage(stage2El);
    activateStage(stage3El);
  }, 2600);

  stageTimers = [t1, t2];
}

function exitLoadingState() {
  loadingSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
}

function clearStageTimers() {
  stageTimers.forEach(clearTimeout);
  stageTimers = [];
}

function activateStage(el) {
  el.classList.remove('is-done');
  el.classList.add('is-active');
}

function doneStage(el) {
  el.classList.remove('is-active');
  el.classList.add('is-done');
}

function setAllStagesDone() {
  [stage1El, stage2El, stage3El].forEach(doneStage);
}

/* ──────────────────────────────────────────────────────────
   Results
   ────────────────────────────────────────────────────────── */
function showResults(data) {
  loadingSection.classList.add('hidden');

  const cfg = CLASS_CONFIG[data.predicted_class] ?? CLASS_CONFIG.notumor;

  /* ── Diagnosis text ── */
  dxClass.textContent       = cfg.label;
  dxClass.style.color       = cfg.color;
  dxDescription.textContent = cfg.description;

  /* ── Images ── */
  resultOriginal.src = objectURL;
  resultHeatmap.src  = `data:image/png;base64,${data.heatmap_image}`;
  if (data.lime_image) {
    resultLime.src = `data:image/png;base64,${data.lime_image}`;
  }

  /* ── Show results section ── */
  resultsSection.classList.remove('hidden');

  // Force reflow so animation re-triggers every time
  void resultsSection.offsetWidth;
  resultsSection.classList.add('animate-in');

  // Scroll results into view on mobile/tablet
  if (window.innerWidth <= 768) {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* ──────────────────────────────────────────────────────────
   New Scan
   ────────────────────────────────────────────────────────── */
newScanBtn.addEventListener('click', () => {
  // Hide results
  resultsSection.classList.add('hidden');
  resultsSection.classList.remove('animate-in');

  // Clear image state
  clearFile();

  // No visuals to reset

  // Show upload section
  uploadSection.classList.remove('hidden');

  // Scroll back to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* ──────────────────────────────────────────────────────────
   Error Toast
   ────────────────────────────────────────────────────────── */
function showError(message) {
  errorMsg.textContent = message;
  errorToast.classList.remove('hidden');

  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(dismissToast, 5000);
}

function dismissToast() {
  errorToast.classList.add('hidden');
  if (toastTimer) { clearTimeout(toastTimer); toastTimer = null; }
}

closeToastBtn.addEventListener('click', dismissToast);
