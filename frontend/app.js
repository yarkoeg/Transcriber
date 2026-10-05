const $ = (sel) => document.querySelector(sel);

const state = {
  jobs: [],
  selected: null,
  result: null,
};

// --- API ---------------------------------------------------------------

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) { /* тело не json — оставляем statusText */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// --- Утилиты -----------------------------------------------------------

const pad = (n) => String(n).padStart(2, '0');

function formatTime(seconds) {
  const total = Math.floor(seconds || 0);
  return `${pad(Math.floor(total / 3600))}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
}

function formatDuration(seconds) {
  if (!seconds) return '';
  const minutes = Math.round(seconds / 60);
  return minutes < 60 ? `${minutes} мин` : `${(minutes / 60).toFixed(1)} ч`;
}

// Отдельно от formatDuration: там длительность записи и округление до минут,
// а здесь время обработки, в котором секунды имеют значение.
function formatElapsed(seconds) {
  const total = Math.round(seconds || 0);
  if (!total) return '';
  if (total < 60) return `${total} с`;
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes < 60) return rest ? `${minutes} мин ${rest} с` : `${minutes} мин`;
  return `${Math.floor(minutes / 60)} ч ${minutes % 60} мин`;
}

function elapsedSince(iso) {
  if (!iso) return 0;
  const started = Date.parse(iso);
  return Number.isNaN(started) ? 0 : (Date.now() - started) / 1000;
}

const STAGE_LABELS = {
  decoding: 'подготовка аудио',
  diarizing: 'разделение говорящих',
  transcribing: 'распознавание',
  merging: 'сборка',
  done: 'готово',
};

function statusLine(job) {
  if (job.status === 'queued') return 'в очереди';
  if (job.status === 'running') return STAGE_LABELS[job.stage] || 'обработка';
  if (job.status === 'failed') return 'ошибка';
  const parts = [formatDuration(job.duration)];
  const meta = job.meta || {};
  // Один говорящий при выключенной диаризации — не результат, а её отсутствие.
  if (meta.diarization_enabled === false) parts.push('без разделения');
  else if (job.speaker_count) parts.push(`${job.speaker_count} говор.`);
  return parts.filter(Boolean).join(' · ');
}

// Время обработки — вторая колонка строки. Живой счётчик перерисовывается сам:
// список и так обновляется раз в секунду.
function timingLine(job) {
  if (job.status === 'running') return formatElapsed(elapsedSince(job.started_at));
  if (job.status === 'queued') return '';
  const elapsed = formatElapsed(job.processing_seconds);
  if (!elapsed) return '';
  return job.status === 'failed' ? elapsed : `за ${elapsed}`;
}

// --- Здоровье сервиса --------------------------------------------------

// Выбор режима запоминается в браузере; пока его нет — берём дефолт из .env.
const MODE_KEY = 'transcriber.diarize';

function storedMode() {
  try {
    return localStorage.getItem(MODE_KEY);
  } catch (_) {
    return null;
  }
}

function setupMode() {
  for (const radio of document.querySelectorAll('input[name="diarize"]')) {
    radio.onchange = () => {
      try { localStorage.setItem(MODE_KEY, radio.value); } catch (_) { /* приватный режим */ }
    };
  }
}

function applyDefaultMode(diarizationEnabled) {
  if (document.querySelector('input[name="diarize"]:checked')) return;
  const value = storedMode() ?? String(Boolean(diarizationEnabled));
  const radio = document.querySelector(`input[name="diarize"][value="${value}"]`);
  if (radio) radio.checked = true;
}

function selectedMode() {
  return document.querySelector('input[name="diarize"]:checked')?.value;
}

async function refreshHealth() {
  try {
    const health = await api('/api/health');
    applyDefaultMode(health.diarization_enabled);
    const problems = [];
    if (!health.ffmpeg) problems.push('нет ffmpeg');
    if (!health.hf_token) problems.push('не задан HF_TOKEN');

    const base = `${health.model} · ${health.asr_device}`;
    $('#health').innerHTML = problems.length
      ? `${base} · <span class="bad">${problems.join(', ')}</span>`
      : base;
  } catch (err) {
    $('#health').innerHTML = '<span class="bad">сервер недоступен</span>';
  }
}

// --- Список задач ------------------------------------------------------

// Отпечаток того, что реально видно в списке. Поллинг идёт раз в секунду,
// и пересборка DOM на каждом тике сбрасывала бы фокус клавиатуры с карточки.
let renderedSignature = '';

function renderJobs() {
  const signature = JSON.stringify([
    state.selected,
    state.jobs.map((job) => [job.id, statusLine(job), timingLine(job), job.progress, job.error]),
  ]);
  if (signature === renderedSignature) return;
  renderedSignature = signature;

  const list = $('#jobs');
  const focusedId = document.activeElement?.closest?.('#jobs')
    ? document.activeElement.dataset.id
    : null;
  list.innerHTML = '';

  for (const job of state.jobs) {
    const row = document.createElement('li');
    // Кнопка, а не li с обработчиком: так карточка доступна с клавиатуры
    // и видна скринридеру.
    const item = document.createElement('button');
    item.className = 'job' + (job.id === state.selected ? ' active' : '');
    item.dataset.id = job.id;
    item.onclick = () => selectJob(job.id);
    row.append(item);

    const name = document.createElement('div');
    name.className = 'job-name';
    name.textContent = job.filename;
    item.append(name);

    const meta = document.createElement('div');
    meta.className = 'job-meta';
    const status = document.createElement('span');
    status.className = job.status;
    status.textContent = statusLine(job);
    meta.append(status);

    const timing = timingLine(job);
    if (timing) {
      // .job-meta уже flex со space-between — второй span уезжает вправо сам.
      const elapsed = document.createElement('span');
      elapsed.className = 'job-elapsed';
      elapsed.textContent = timing;
      meta.append(elapsed);
    }
    item.append(meta);

    if (job.status === 'running' || job.status === 'queued') {
      const bar = document.createElement('div');
      bar.className = 'bar';
      const fill = document.createElement('div');
      fill.style.width = `${Math.round((job.progress || 0) * 100)}%`;
      bar.append(fill);
      item.append(bar);
    }

    if (job.status === 'failed' && job.error) {
      const error = document.createElement('div');
      error.className = 'job-error';
      error.textContent = job.error;
      item.append(error);
    }

    list.append(row);
    if (job.id === focusedId) item.focus();
  }
}

async function refreshJobs() {
  try {
    const { jobs } = await api('/api/jobs');
    const previous = new Map(state.jobs.map((job) => [job.id, job.status]));
    state.jobs = jobs;
    renderJobs();

    // Задача только что завершилась и открыта — подтянуть результат.
    const current = jobs.find((job) => job.id === state.selected);
    if (current && current.status === 'done' && previous.get(current.id) !== 'done') {
      await openResult(current);
    }
  } catch (err) {
    // Сервер мог перезапускаться — молча ждём следующего тика.
  }
}

// --- Просмотр результата -----------------------------------------------

// textContent, а не innerHTML: в тексте бывает вывод ffmpeg и имя файла.
function showMessage(text) {
  const message = document.createElement('p');
  message.className = 'empty';
  message.textContent = text;
  $('#viewer').replaceChildren(message);
}

async function selectJob(jobId) {
  state.selected = jobId;
  renderJobs();
  const job = state.jobs.find((item) => item.id === jobId);
  if (!job) return;

  if (job.status === 'done') {
    await openResult(job);
  } else if (job.status === 'failed') {
    showMessage(`Обработка не удалась: ${job.error || 'неизвестная ошибка'}`);
  } else {
    showMessage('Идёт обработка…');
  }
}

async function openResult(job) {
  try {
    state.result = await api(`/api/jobs/${job.id}/result`);
  } catch (err) {
    showMessage(`Не удалось загрузить результат: ${err.message}`);
    return;
  }
  renderResult(job);
}

function renderResult(job) {
  const viewer = $('#viewer');
  viewer.innerHTML = '';

  const head = document.createElement('div');
  head.className = 'viewer-head';

  const title = document.createElement('h2');
  title.textContent = job.filename;
  head.append(title);

  const info = document.createElement('span');
  info.className = 'viewer-info';
  const meta = state.result.meta || {};
  info.textContent = [
    formatDuration(meta.duration),
    `${state.result.speakers.length} говорящих`,
    meta.pipeline_seconds ? `обработано за ${formatElapsed(meta.pipeline_seconds)}` : '',
    meta.realtime_factor ? `${meta.realtime_factor}× быстрее записи` : '',
    meta.model,
  ].filter(Boolean).join(' · ');
  // Разбивка по стадиям нужна при разборе «почему так долго», но в самой строке
  // была бы шумом — прячем в подсказку. Секунды сырые: стадия в 0.4 с иначе
  // схлопнулась бы в пустую строку.
  info.title = Object.entries(meta.stage_seconds || {})
    .map(([stage, seconds]) => `${STAGE_LABELS[stage] || stage}: ${seconds} с`)
    .join('\n');
  head.append(info);

  const spacer = document.createElement('span');
  spacer.style.flex = '1';
  head.append(spacer);

  for (const fmt of ['txt', 'json']) {
    const button = document.createElement('button');
    button.textContent = `Скачать .${fmt}`;
    button.onclick = () => window.location.assign(`/api/jobs/${job.id}/download?fmt=${fmt}`);
    head.append(button);
  }

  const remove = document.createElement('button');
  remove.className = 'danger';
  remove.textContent = 'Удалить';
  remove.onclick = () => deleteJob(job.id);
  head.append(remove);

  viewer.append(head);

  const player = document.createElement('audio');
  player.controls = true;
  player.preload = 'none';
  player.src = `/api/jobs/${job.id}/audio`;
  viewer.append(player);

  const container = document.createElement('div');
  container.className = 'utterances';

  for (const utterance of state.result.utterances) {
    const row = document.createElement('div');
    row.className = 'utt';
    row.dataset.start = utterance.start;
    row.dataset.end = utterance.end;

    const headCell = document.createElement('div');
    headCell.className = 'utt-head';

    const speaker = document.createElement('button');
    speaker.className = 'utt-speaker';
    speaker.textContent = utterance.speaker;
    speaker.title = 'Нажмите, чтобы переименовать';
    speaker.onclick = () => renameSpeaker(job.id, utterance.speaker);
    headCell.append(speaker);

    const time = document.createElement('button');
    time.className = 'utt-time';
    time.textContent = formatTime(utterance.start);
    time.onclick = () => {
      player.currentTime = utterance.start;
      player.play();
    };
    headCell.append(time);

    const text = document.createElement('div');
    text.className = 'utt-text';
    text.textContent = utterance.text;

    row.append(headCell, text);
    container.append(row);
  }

  viewer.append(container);

  player.ontimeupdate = () => highlight(container, player.currentTime);
}

let highlighted = null;

// Реплики идут по времени, поэтому бинарный поиск: timeupdate приходит
// несколько раз в секунду, а в двухчасовой записи реплик тысячи.
function highlight(container, time) {
  const rows = container.children;
  let low = 0;
  let high = rows.length - 1;
  let row = null;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const item = rows[mid];
    if (time < Number(item.dataset.start)) high = mid - 1;
    else if (time > Number(item.dataset.end)) low = mid + 1;
    else { row = item; break; }
  }
  if (row === highlighted) return;
  if (highlighted) highlighted.classList.remove('playing');
  if (row) row.classList.add('playing');
  highlighted = row;
}

async function renameSpeaker(jobId, currentName) {
  const next = window.prompt(`Как назвать «${currentName}»?`, currentName);
  if (!next || next === currentName) return;

  try {
    state.result = await api(`/api/jobs/${jobId}/speakers`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names: { [currentName]: next } }),
    });
  } catch (err) {
    window.alert(`Не удалось переименовать: ${err.message}`);
    return;
  }

  const job = state.jobs.find((item) => item.id === jobId);
  if (job) renderResult(job);
}

async function deleteJob(jobId) {
  if (!window.confirm('Удалить запись и её расшифровку?')) return;
  try {
    await api(`/api/jobs/${jobId}`, { method: 'DELETE' });
  } catch (err) {
    window.alert(`Не удалось удалить: ${err.message}`);
    return;
  }
  state.selected = null;
  state.result = null;
  showMessage('Выберите запись слева или загрузите новую.');
  await refreshJobs();
}

// --- Загрузка файлов ---------------------------------------------------

async function upload(files) {
  for (const file of files) {
    const form = new FormData();
    form.append('file', file);
    // Не выбрано (сервер ещё не ответил) — сервер возьмёт значение из .env.
    const mode = selectedMode();
    if (mode) form.append('diarize', mode);
    try {
      await api('/api/jobs', { method: 'POST', body: form });
    } catch (err) {
      window.alert(`${file.name}: ${err.message}`);
    }
  }
  await refreshJobs();
}

function setupDropzone() {
  const drop = $('#drop');
  const input = $('#file');

  drop.onclick = () => input.click();
  input.onchange = () => {
    if (input.files.length) upload([...input.files]);
    input.value = '';
  };

  for (const type of ['dragenter', 'dragover']) {
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add('over');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.remove('over');
    });
  }
  drop.addEventListener('drop', (event) => {
    if (event.dataTransfer?.files?.length) upload([...event.dataTransfer.files]);
  });
}

// --- Старт -------------------------------------------------------------

setupDropzone();
setupMode();
refreshHealth();
refreshJobs();
setInterval(refreshJobs, 1000);
setInterval(refreshHealth, 10000);
