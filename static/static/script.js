const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const queueInfo = document.getElementById('queueInfo');
const resultsSection = document.getElementById('results');
const grid = document.getElementById('grid');
const downloadAllBtn = document.getElementById('downloadAllBtn');

let queuedFiles = [];
let processedNames = [];

function refreshQueueUI() {
  if (queuedFiles.length === 0) {
    queueInfo.classList.add('hidden');
    queueInfo.innerHTML = '';
    processBtn.disabled = true;
    return;
  }
  queueInfo.classList.remove('hidden');
  queueInfo.innerHTML = `<strong>${queuedFiles.length} resim seçildi</strong>`;
  processBtn.disabled = false;
}

function addFiles(fileList) {
  for (const f of fileList) {
    if (f.type.startsWith('image/')) queuedFiles.push(f);
  }
  refreshQueueUI();
}

dropzone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'INPUT') fileInput.click();
});
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', (e) => {
  addFiles(e.target.files);
  fileInput.value = '';
});

processBtn.addEventListener('click', async () => {
  if (queuedFiles.length === 0) return;
  processBtn.disabled = true;
  processBtn.textContent = 'İşleniyor...';

  const langs = Array.from(document.querySelectorAll('input[name="lang"]:checked')).map(i => i.value);
  const formData = new FormData();
  for (const f of queuedFiles) formData.append('images', f, f.name);
  for (const l of langs) formData.append('langs', l);

  try {
    const resp = await fetch('/api/process', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    renderResults(data.results);
  } catch (err) {
    alert('İşlem sırasında hata oluştu: ' + err);
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = 'Balonları Temizle';
    queuedFiles = [];
    refreshQueueUI();
  }
});

function renderResults(results) {
  grid.innerHTML = '';
  processedNames = [];
  resultsSection.classList.remove('hidden');
  for (const r of results) {
    const card = document.createElement('div');
    card.className = 'card';
    if (r.error) {
      card.innerHTML = `<div class="card-error">${r.filename}: ${r.error}</div>`;
    } else {
      processedNames.push(r.filename);
      card.innerHTML = `
        <img src="data:image/jpeg;base64,${r.preview}" alt="${r.filename}">
        <div class="card-body">
          <div class="card-name">${r.filename}</div>
          <a class="card-dl" href="/api/download/${encodeURIComponent(r.filename)}" download>İndir</a>
        </div>`;
    }
    grid.appendChild(card);
  }
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

downloadAllBtn.addEventListener('click', () => {
  if (processedNames.length === 0) return;
  const params = processedNames.map(n => 'f=' + encodeURIComponent(n)).join('&');
  window.location.href = '/api/download_all?' + params;
});
