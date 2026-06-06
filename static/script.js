const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const browseBtn      = document.getElementById('browseBtn');
const predictBtn     = document.getElementById('predictBtn');
const btnText        = document.getElementById('btnText');
const spinner        = document.getElementById('spinner');

const dropIcon       = document.getElementById('dropIcon');
const dropText       = document.getElementById('dropText');
const dropPreview    = document.getElementById('dropPreview');
const previewImg     = document.getElementById('previewImg');

const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent     = document.getElementById('resultContent');
const resultClass       = document.getElementById('resultClass');
const resultIcon        = document.getElementById('resultIcon');
const confidenceBadge   = document.getElementById('confidenceBadge');
const probList          = document.getElementById('probList');

const CLASS_ICONS = {
  banana: '🍌', guava: '🍈', jackfruit: '🌳', mango: '🥭', neem: '🌿'
};

let selectedFile = null;

// ── File Selection ──────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
  if (e.target === dropZone || e.target.closest('.drop-icon') || e.target.closest('.drop-text')) {
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Drag & Drop ─────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

// ── Handle File ─────────────────────────────────────────────────────────────
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropIcon.style.display = 'none';
    dropText.style.display = 'none';
    dropPreview.style.display = 'flex';
  };
  reader.readAsDataURL(file);
  predictBtn.disabled = false;

  // Reset results
  resultPlaceholder.style.display = 'block';
  resultContent.style.display = 'none';
}

// ── Predict ─────────────────────────────────────────────────────────────────
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Loading state
  predictBtn.disabled = true;
  btnText.textContent = 'Đang phân tích...';
  spinner.style.display = 'inline-block';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      alert('Lỗi: ' + data.error);
      return;
    }

    renderResult(data);
  } catch (err) {
    alert('Lỗi kết nối tới server: ' + err.message);
  } finally {
    predictBtn.disabled = false;
    btnText.textContent = 'Phân tích lại';
    spinner.style.display = 'none';
  }
});

// ── Render Result ───────────────────────────────────────────────────────────
function renderResult(data) {
  const cls  = data.class;
  const conf = data.confidence;

  resultClass.textContent      = cls;
  confidenceBadge.textContent  = conf + '%';
  resultIcon.textContent       = CLASS_ICONS[cls] || '🌿';

  // Determine confidence color
  if (conf >= 70) confidenceBadge.style.background = 'linear-gradient(135deg,#4ade80,#22d3ee)';
  else if (conf >= 40) confidenceBadge.style.background = 'linear-gradient(135deg,#facc15,#fb923c)';
  else confidenceBadge.style.background = 'linear-gradient(135deg,#f87171,#fb923c)';

  // Probability bars – sorted descending
  const sorted = Object.entries(data.all_probs).sort((a, b) => b[1] - a[1]);

  probList.innerHTML = '';
  sorted.forEach(([name, prob], idx) => {
    const isTop = idx === 0;
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.style.animationDelay = (idx * 0.06) + 's';
    item.innerHTML = `
      <span class="prob-name">${CLASS_ICONS[name] || '🌿'} ${name}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${isTop ? '' : 'muted-bar'}" data-val="${prob}"></div>
      </div>
      <span class="prob-val">${prob}%</span>
    `;
    probList.appendChild(item);
  });

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll('.prob-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.val + '%';
    });
  });

  resultPlaceholder.style.display = 'none';
  resultContent.style.display     = 'block';
}
