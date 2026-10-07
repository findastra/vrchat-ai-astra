const state = {
  bootstrap: null,
  manifest: null,
  session: null,
  service: null,
  settings: null,
  featureRuntime: null,
  logs: [],
  conversation: [],
  selectedTrainingPaths: [],
  trainingPlans: [],
  trainingResult: null,
  backendState: 'idle',
  feedbackBusy: false,
};

const elements = {};

function formatNumber(value) {
  const number = Number(value || 0);
  return new Intl.NumberFormat().format(number);
}

function formatPercent(value, digits = 0) {
  const number = Number(value || 0);
  return `${(number * 100).toFixed(digits)}%`;
}

function formatMaybeNumber(value, digits = 1, suffix = '') {
  if (value === null || value === undefined || value === '') {
    return `0${suffix}`;
  }
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function formatOptionalPercent(value, digits = 0, fallback = 'n/a') {
  if (value === null || value === undefined || value === '') {
    return fallback;
  }
  return formatPercent(value, digits);
}

function titleCaseWords(value, fallback = 'Unknown') {
  const text = String(value || '').trim();
  if (!text) {
    return fallback;
  }
  return text
    .replace(/_/g, ' ')
    .split(/\s+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function formatHardwareAccelerator(hardware) {
  const activeGpu = (hardware && hardware.active_gpu) || {};
  if (!activeGpu || activeGpu.gpu_available === false) {
    return 'CPU only';
  }
  const gpuType = titleCaseWords(activeGpu.gpu_type || '', '');
  const deviceName = String(activeGpu.device_name || '').trim();
  return [gpuType, deviceName].filter(Boolean).join(' / ') || 'GPU active';
}

function formatPreferredGpu(hardware) {
  const recommended = (hardware && hardware.recommended_gpu) || {};
  const gpuName = String(recommended.name || '').trim();
  if (gpuName) {
    return gpuName;
  }
  const gpuIndex = hardware && hardware.recommended_gpu_index;
  if (gpuIndex !== null && gpuIndex !== undefined && gpuIndex !== '') {
    return `GPU ${gpuIndex}`;
  }
  return 'Auto';
}

function escapeHtml(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function normalizePathText(value) {
  return String(value || '').replace(/\//g, '\\');
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getPathRoots() {
  const service = state.service || {};
  return [service.workspace_root, service.ai_astra_root]
    .filter(Boolean)
    .map((root) => normalizePathText(root).replace(/[\\]+$/, ''));
}

function getRelativePath(value) {
  const normalized = normalizePathText(value);
  if (!normalized) {
    return '';
  }
  const lowerPath = normalized.toLowerCase();
  for (const root of getPathRoots()) {
    const lowerRoot = root.toLowerCase();
    if (lowerPath === lowerRoot) {
      return '.';
    }
    if (lowerPath.startsWith(`${lowerRoot}\\`)) {
      return normalized.slice(root.length + 1);
    }
  }
  return normalized;
}

function relativeText(value) {
  let normalized = normalizePathText(value);
  for (const root of getPathRoots()) {
    const pattern = new RegExp(`${escapeRegExp(root)}(?:\\\\)?`, 'gi');
    normalized = normalized.replace(pattern, '');
  }
  return normalized.replace(/^\\+/, '');
}

function unwrapEnvelope(payload) {
  if (!payload) {
    throw new Error('Empty response from backend.');
  }
  if (payload.ok === false) {
    throw new Error(payload.error || 'Backend request failed.');
  }
  return Object.prototype.hasOwnProperty.call(payload, 'result') ? payload.result : payload;
}

function batchMap(batchPayload) {
  const result = {};
  const items = Array.isArray(batchPayload && batchPayload.results) ? batchPayload.results : [];
  for (const item of items) {
    result[item.id] = item;
  }
  return result;
}

function setAlert(message, tone = 'info') {
  elements.alertBar.textContent = message;
  elements.alertBar.className = `alert-bar alert-bar--${tone}`;
}

function setBusy(button, busy, busyLabel) {
  if (!button) {
    return;
  }
  if (!button.dataset.defaultLabel) {
    button.dataset.defaultLabel = button.textContent;
  }
  button.disabled = busy;
  button.textContent = busy ? busyLabel : button.dataset.defaultLabel;
}

function activateTab(tabName) {
  for (const button of document.querySelectorAll('[data-tab]')) {
    const isActive = button.dataset.tab === tabName;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-selected', String(isActive));
    button.tabIndex = isActive ? 0 : -1;
  }
  for (const panel of document.querySelectorAll('[data-panel]')) {
    const isActive = panel.dataset.panel === tabName;
    panel.classList.toggle('is-active', isActive);
    panel.setAttribute('aria-hidden', String(!isActive));
  }
}

function buildConversationArchive() {
  return state.conversation
    .map((entry) => `${entry.role.toUpperCase()}: ${entry.text}`)
    .join('\n\n');
}

function lastAssistantMessage() {
  return [...state.conversation].reverse().find((entry) => entry.role === 'assistant') || null;
}

function renderChrome() {
  const hasService = Boolean(state.service && state.service.base_url);
  const baseUrl = state.backendState === 'running'
    ? (hasService ? state.service.base_url : 'Connecting...')
    : 'Backend offline';
  const sessionId = state.session && state.session.session_id ? state.session.session_id.slice(0, 8) : '--';
  const transportTone = state.backendState === 'running' && hasService ? 'ONLINE' : state.backendState === 'running' ? 'STARTING' : state.backendState.toUpperCase();
  elements.transportState.textContent = transportTone;
  elements.sessionId.textContent = `Session ${sessionId}`;
  elements.serviceUrl.textContent = baseUrl;
  elements.railState.textContent = state.backendState;
  elements.railBackend.textContent = baseUrl;
  elements.methodCount.textContent = formatNumber((state.manifest && state.manifest.transport && state.manifest.transport.method_count) || 0);
}

function renderMetrics() {
  const bootstrap = state.bootstrap || {};
  const generation = bootstrap.generation || {};
  const memory = bootstrap.memory || {};
  const status = bootstrap.status || {};
  const hardware = bootstrap.hardware || {};
  const featureRuntime = bootstrap.feature_runtime || state.featureRuntime || {};
  const knowledge = bootstrap.knowledge || featureRuntime.knowledge_stats || {};
  const learningHealth = featureRuntime.adaptive_learning_health || {};
  const qualityScore = Number(generation.recent_quality || status.recent_quality || 0);
  const memoryPercent = Number(memory.memory_usage_percent || 0);

  elements.qualityGauge.style.setProperty('--fill', String(Math.max(0, Math.min(qualityScore, 1))));
  elements.qualityGaugeValue.textContent = formatPercent(qualityScore);
  elements.qualityGaugeCaption.textContent = generation.generation_stats || 'No responses yet';

  elements.memoryBar.style.width = `${Math.max(0, Math.min(memoryPercent * 100, 100))}%`;
  elements.memoryValue.textContent = `${formatPercent(memoryPercent)} / ${formatMaybeNumber(memory.memory_usage_mb, 1, ' MB')}`;
  elements.chainCount.textContent = formatNumber(generation.chain_count);
  elements.associationCount.textContent = formatNumber(generation.association_count);
  elements.clusterCount.textContent = formatNumber(status.cluster_count || generation.cluster_count);
  elements.knowledgeFactCount.textContent = formatNumber(knowledge.fact_count);
  elements.storageMode.textContent = (bootstrap.storage && bootstrap.storage.label) || 'Unknown';
  elements.generationStats.textContent = generation.generation_stats || 'No generation data';
  elements.conversationCount.textContent = formatNumber(status.conversation_memory_count);
  elements.availableMemory.textContent = formatMaybeNumber(memory.available_memory_mb, 1, ' MB');
  elements.controlMethodCount.textContent = formatNumber((state.manifest && state.manifest.transport && state.manifest.transport.control_methods && state.manifest.transport.control_methods.length) || 0);
  elements.knowledgeConceptCount.textContent = formatNumber(knowledge.concept_count);
  elements.knowledgeEvidenceCount.textContent = formatNumber(knowledge.evidence_count);
  elements.learningHealthMode.textContent = titleCaseWords(learningHealth.mode, learningHealth.active === false ? 'Inactive' : 'Balanced');
  elements.learningHealthAcceptance.textContent = formatOptionalPercent(learningHealth.recent_learning_acceptance_rate, 0);
  elements.learningHealthTrend.textContent = titleCaseWords(learningHealth.recent_quality_trend, 'Stable');
  elements.learningHealthGate.textContent = formatOptionalPercent(learningHealth.recommended_min_quality, 0);
  elements.hardwareTier.textContent = titleCaseWords(hardware.tier);
  elements.hardwareAccelerator.textContent = formatHardwareAccelerator(hardware);
  elements.hardwarePreferredGpu.textContent = formatPreferredGpu(hardware);
  elements.hardwareWorkers.textContent = formatNumber(hardware.recommended_parallel_workers);

  const vocab = bootstrap.vocab_id_storage || {};
  elements.vocabIdMode.textContent = String(vocab.mode || 'off');
  elements.vocabLexiconSize.textContent = formatNumber(vocab.lexicon_size);
  elements.vocabIdChainCount.textContent = formatNumber(vocab.id_chain_count);
  elements.vocabTextChainCount.textContent = formatNumber(vocab.text_chain_count);

  const activeFeatures = featureRuntime.active_feature_flags || [];
  if (!activeFeatures.length) {
    elements.featureCloud.innerHTML = '<span class="feature-pill">No optional features active</span>';
    return;
  }
  elements.featureCloud.innerHTML = activeFeatures
    .map((name) => `<span class="feature-pill">${escapeHtml(name.replace(/_/g, ' '))}</span>`)
    .join('');
}

function renderConversation() {
  if (!state.conversation.length) {
    elements.transcript.innerHTML = '<div class="transcript__empty">AI Astra is ready. Send a message to start the conversation.</div>';
    elements.responseStats.textContent = 'No response yet.';
    elements.rewardButton.disabled = true;
    elements.penalizeButton.disabled = true;
    return;
  }

  elements.transcript.innerHTML = state.conversation
    .map((entry) => {
      const meta = entry.meta ? `<div class="message__meta">${escapeHtml(entry.meta)}</div>` : '';
      return `
        <article class="message message--${escapeHtml(entry.role)}">
          <p class="message__role">${escapeHtml(entry.role)}</p>
          <p class="message__body">${escapeHtml(entry.text)}</p>
          ${meta}
        </article>
      `;
    })
    .join('');

  const lastAssistant = lastAssistantMessage();
  elements.responseStats.textContent = lastAssistant && lastAssistant.meta ? lastAssistant.meta : 'Waiting for the next response.';
  const feedbackDisabled = !lastAssistant || state.feedbackBusy;
  elements.rewardButton.disabled = feedbackDisabled;
  elements.penalizeButton.disabled = feedbackDisabled;
  elements.transcript.scrollTop = elements.transcript.scrollHeight;
}

function renderTraining() {
  const extensions = (((state.manifest || {}).training_extensions) || []).join(', ');
  elements.trainingExtensions.textContent = extensions ? `Accepted extensions: ${extensions}` : 'Accepted extensions: backend default';

  if (!state.selectedTrainingPaths.length) {
    elements.trainingPathList.innerHTML = '<div class="path-item"><p class="path-item__name">No training sources queued</p><p class="path-item__path">Use "Select Sources" to choose files or folders.</p></div>';
  } else {
    elements.trainingPathList.innerHTML = state.selectedTrainingPaths
      .map((filePath) => `
        <div class="path-item">
          <p class="path-item__name">${escapeHtml(filePath.split(/[/\\]/).pop() || filePath)}</p>
          <p class="path-item__path">${escapeHtml(getRelativePath(filePath))}</p>
        </div>
      `)
      .join('');
  }

  if (!state.trainingPlans.length) {
    elements.trainingPlanList.innerHTML = '<div class="plan-card"><p class="path-item__name">No plan loaded</p><p class="plan-card__meta">Queue sources to calculate chunk counts, worker lanes, and GPU usage.</p></div>';
  } else {
    elements.trainingPlanList.innerHTML = state.trainingPlans
      .map((plan) => `
        <div class="plan-card">
          <p class="path-item__name">${escapeHtml(plan.file_path.split(/[/\\]/).pop() || plan.file_path)}</p>
          <p class="path-item__path">${escapeHtml(getRelativePath(plan.file_path))}</p>
          <p class="plan-card__meta">
            ${escapeHtml(`Words ${formatNumber(plan.estimated_word_count)} | Chunks ${formatNumber(plan.estimated_chunks)} | Chunk size ${formatNumber(plan.chunk_size)} | Workers ${formatNumber(plan.parallel_workers)}`)}
          </p>
        </div>
      `)
      .join('');
  }

  elements.trainingResult.textContent = state.trainingResult || 'Queue a file or folder to inspect AI Astra\'s training plan.';
}

function renderSettings() {
  const settings = state.settings || {};
  elements.performanceProfile.value = String(settings.performance_profile || 'max');
  elements.maxResponseLength.value = Number(settings.max_response_length || 25);
  elements.creativityFactor.value = Number(settings.creativity_factor || 0.49);
  elements.qualityThreshold.value = Number(settings.quality_threshold || 0.41);
  elements.enhancedIntelligence.checked = Boolean(settings.enhanced_intelligence);
  elements.advancedReasoning.checked = Boolean(settings.advanced_reasoning);
  elements.adaptiveLearning.checked = Boolean(settings.adaptive_learning);
  elements.hardwareAdaptiveMode.checked = Boolean(settings.hardware_adaptive_mode);
}

function renderDocs() {
  const workflows = (((state.manifest || {}).transport || {}).frontend_workflows) || {};
  const workflowNames = Object.keys(workflows);
  if (!workflowNames.length) {
    elements.workflowList.innerHTML = '<div class="workflow-card"><p class="workflow-card__title">No workflow hints published</p></div>';
    return;
  }
  elements.workflowList.innerHTML = workflowNames
    .map((name) => `
      <div class="workflow-card">
        <p class="workflow-card__title">${escapeHtml(name.replace(/_/g, ' '))}</p>
        <p class="workflow-card__copy">${escapeHtml((workflows[name] || []).join(' -> '))}</p>
      </div>
    `)
    .join('');
}

function renderLogs() {
  if (!state.logs.length) {
    elements.logFeed.innerHTML = '<div class="log-entry"><div class="log-entry__head"><span>Backend</span><span>Idle</span></div><p class="log-entry__body">No backend activity yet.</p></div>';
    return;
  }
  elements.logFeed.innerHTML = state.logs
    .slice(-120)
    .map((entry) => `
      <article class="log-entry log-entry--${escapeHtml(entry.stream)}">
        <div class="log-entry__head">
          <span>${escapeHtml(entry.stream)}</span>
          <span>${escapeHtml(entry.timestamp.split('T')[1].replace('Z', ''))}</span>
        </div>
        <p class="log-entry__body">${escapeHtml(relativeText(entry.text))}</p>
      </article>
    `)
    .join('');
  elements.logFeed.scrollTop = elements.logFeed.scrollHeight;
}

function renderAll() {
  renderChrome();
  renderMetrics();
  renderConversation();
  renderTraining();
  renderSettings();
  renderDocs();
  renderLogs();
}

async function refreshRuntime() {
  const batchPayload = await window.aiAstraBridge.batch([
    { id: 'bootstrap', method: 'get_runtime_bootstrap_snapshot', params: {} },
    { id: 'features', method: 'get_feature_runtime_snapshot', params: {} },
    { id: 'settings', method: 'get_settings_snapshot', params: {} },
    { id: 'session', method: 'get_session_info', params: {} },
  ]);
  const results = batchMap(batchPayload);
  state.bootstrap = unwrapEnvelope(results.bootstrap);
  state.featureRuntime = unwrapEnvelope(results.features);
  state.settings = unwrapEnvelope(results.settings);
  state.session = unwrapEnvelope(results.session);
  renderAll();
}

async function bootstrap() {
  try {
    const payload = await window.aiAstraBridge.bootstrap();
    state.backendState = 'running';
    state.bootstrap = payload.startup.bootstrap;
    state.featureRuntime = payload.startup.features;
    state.settings = payload.startup.settings;
    state.manifest = payload.manifest;
    state.session = payload.session;
    state.service = payload.service;
    state.logs = payload.logs || [];
    renderAll();
    setAlert('Standalone app connected to the live backend.', 'success');
  } catch (error) {
    state.backendState = 'error';
    renderChrome();
    setAlert(`Startup failed: ${error.message}`, 'error');
  }
}

function resetProvenance() {
  if (!elements.provenanceGoal) return;
  elements.provenanceGoal.textContent = '--';
  elements.provenanceAction.textContent = '--';
  elements.provenanceDirective.textContent = '--';
  elements.provenanceMode.textContent = '--';
  elements.provenanceHygiene.textContent = '--';
  elements.provenanceEvidence.textContent = '--';
  if (elements.provenanceContinuity) elements.provenanceContinuity.textContent = '--';
  if (elements.provenanceEpisode) elements.provenanceEpisode.textContent = '--';
}

function updateProvenance(result) {
  if (!elements.provenanceGoal) return;
  const metadata = result?.metadata || {};
  const cognition = metadata.cognitive_trace || metadata.response_plan?.cognitive_control || {};
  const realization = metadata.realization_trace || {};
  const hygiene = metadata.learning_hygiene || {};
  const plan = metadata.response_plan || {};
  const episode = cognition.episode_memory || {};
  const mainClaim = plan.main_claim?.text || plan.supporting_facts?.[0]?.summary || '';
  elements.provenanceGoal.textContent = cognition.goal ? String(cognition.goal).replace(/_/g, ' ') : '--';
  elements.provenanceAction.textContent = cognition.action ? String(cognition.action).replace(/_/g, ' ') : '--';
  elements.provenanceDirective.textContent = cognition.directive || '--';
  const modeParts = [realization.mode || '--'];
  if (realization.preempted) modeParts.push('preempted');
  if (realization.cognitive_action) modeParts.push(String(realization.cognitive_action).replace(/_/g, ' '));
  elements.provenanceMode.textContent = modeParts.join(' / ');
  if (typeof hygiene.accept === 'boolean') {
    elements.provenanceHygiene.textContent = `${hygiene.accept ? 'accept' : 'reject'} (${hygiene.reason || 'n/a'})`;
  } else {
    elements.provenanceHygiene.textContent = '--';
  }
  elements.provenanceEvidence.textContent = mainClaim
    ? String(mainClaim).slice(0, 160)
    : (Array.isArray(realization.used_texts) && realization.used_texts[0]
      ? String(realization.used_texts[0]).slice(0, 160)
      : '--');
  if (elements.provenanceContinuity) {
    elements.provenanceContinuity.textContent = cognition.goal_continuity ? 'continuing prior goal' : 'fresh selection';
  }
  if (elements.provenanceEpisode) {
    const preview = episode.bot_preview || episode.user_preview || '';
    elements.provenanceEpisode.textContent = preview
      ? `score ${episode.score ?? '--'}: ${String(preview).slice(0, 120)}`
      : '--';
  }
}

async function handleSend(event) {
  event.preventDefault();
  const prompt = elements.promptInput.value.trim();
  if (!prompt) {
    return;
  }

  elements.promptInput.value = '';
  state.conversation.push({ role: 'user', text: prompt });
  renderConversation();
  resetProvenance();
  setBusy(elements.sendButton, true, 'Sending');
  setAlert('Sending message to the live backend...', 'info');

  try {
    const result = unwrapEnvelope(await window.aiAstraBridge.invoke('generate_response', { user_input: prompt }));
    const cognition = result.metadata?.cognitive_trace || {};
    const cognitiveText = cognition.goal && cognition.action
      ? ` | Goal ${String(cognition.goal).replace(/_/g, ' ')} | Strategy ${String(cognition.action).replace(/_/g, ' ')}`
      : '';
    const qualityText = `Quality ${formatPercent(result.quality_score || 0)} | ${result.generation_stats || 'No stats'}${cognitiveText}`;
    state.conversation.push({
      role: 'assistant',
      text: result.response || '[No response]',
      meta: qualityText,
    });
    updateProvenance(result);
    await refreshRuntime();
    renderConversation();
    setAlert('Response complete.', 'success');
  } catch (error) {
    state.conversation.push({ role: 'system', text: `Request failed: ${error.message}` });
    renderConversation();
    resetProvenance();
    setAlert(`Response failed: ${error.message}`, 'error');
  } finally {
    setBusy(elements.sendButton, false);
  }
}

async function applyFeedback(isPositive) {
  const assistant = lastAssistantMessage();
  if (!assistant || state.feedbackBusy) {
    if (!assistant) {
      setAlert('There is no assistant response to score yet.', 'error');
    }
    return;
  }
  state.feedbackBusy = true;
  setBusy(elements.rewardButton, true, 'Saving');
  setBusy(elements.penalizeButton, true, 'Saving');
  try {
    const payload = unwrapEnvelope(await window.aiAstraBridge.invoke('apply_feedback', {
      sentence: assistant.text,
      is_positive: isPositive,
    }));
    await refreshRuntime();
    setAlert(payload.message || 'Feedback recorded.', 'success');
  } catch (error) {
    setAlert(`Feedback failed: ${error.message}`, 'error');
  } finally {
    state.feedbackBusy = false;
    setBusy(elements.rewardButton, false);
    setBusy(elements.penalizeButton, false);
    renderConversation();
  }
}

async function exportConversation() {
  if (!state.conversation.length) {
    setAlert('No conversation is loaded for export.', 'error');
    return;
  }
  setBusy(elements.exportConversationButton, true, 'Exporting');
  try {
    const payload = unwrapEnvelope(await window.aiAstraBridge.invoke('export_conversation_text', {
      text: buildConversationArchive(),
      folder_name: 'conversations',
    }));
    if (payload.path) {
      setAlert(`Chat exported to ${getRelativePath(payload.path)}.`, 'success');
    } else {
      setAlert(payload.message || 'Chat exported.', 'success');
    }
    if (payload.path) {
      await window.aiAstraBridge.openPath(payload.path);
    }
  } catch (error) {
    setAlert(`Export failed: ${error.message}`, 'error');
  } finally {
    setBusy(elements.exportConversationButton, false);
  }
}

async function pickTrainingSources() {
  const selection = await window.aiAstraBridge.pickTrainingSources();
  if (!selection || selection.canceled) {
    return;
  }
  state.selectedTrainingPaths = selection.filePaths || [];
  setAlert(`Queued ${state.selectedTrainingPaths.length} training source(s).`, 'info');
  await recalcTrainingPlans();
}

async function recalcTrainingPlans() {
  if (!state.selectedTrainingPaths.length) {
    state.trainingPlans = [];
    state.trainingResult = 'Queue one or more files or folders to calculate a training plan.';
    renderTraining();
    return;
  }

  setBusy(elements.refreshTrainingButton, true, 'Scanning');
  try {
    const discovery = unwrapEnvelope(await window.aiAstraBridge.invoke('collect_training_files', {
      paths: state.selectedTrainingPaths,
    }));
    const requests = (discovery.paths || []).slice(0, 6).map((filePath, index) => ({
      id: `plan-${index}`,
      method: 'get_training_plan',
      params: { file_path: filePath },
    }));
    const batchPayload = requests.length ? await window.aiAstraBridge.batch(requests) : { results: [] };
    const results = batchMap(batchPayload);
    state.trainingPlans = requests
      .map((request) => unwrapEnvelope(results[request.id]))
      .filter(Boolean);
    state.trainingResult = `Resolved ${formatNumber(discovery.count)} trainable file(s). Showing plans for the first ${formatNumber(state.trainingPlans.length)} source(s).`;
    renderTraining();
    setAlert('Training plans recalculated.', 'success');
  } catch (error) {
    state.trainingPlans = [];
    state.trainingResult = `Training plan failed: ${error.message}`;
    renderTraining();
    setAlert(`Training plan failed: ${error.message}`, 'error');
  } finally {
    setBusy(elements.refreshTrainingButton, false);
  }
}

async function runTraining() {
  if (!state.selectedTrainingPaths.length) {
    setAlert('Queue training sources before starting a run.', 'error');
    return;
  }

  const chunkSizeOverride = Number(elements.chunkSizeOverride.value || 0);
  const priorityBoost = Number(elements.priorityBoost.value || 2);

  setBusy(elements.runTrainingButton, true, 'Training');
  setAlert('Training is running through the backend API...', 'info');
  try {
    const payload = unwrapEnvelope(await window.aiAstraBridge.invoke('train_files', {
      paths: state.selectedTrainingPaths,
      chunk_size_override: chunkSizeOverride > 0 ? chunkSizeOverride : null,
      base_priority_boost: priorityBoost > 0 ? priorityBoost : 2,
    }));
    const lines = [
      payload.message || 'Training completed.',
      `Successes: ${formatNumber(payload.success_count)} | Failures: ${formatNumber(payload.failure_count)}`,
      `Words processed: ${formatNumber(payload.total_words_processed)}`,
      `Knowledge facts extracted: ${formatNumber(payload.knowledge_fact_count)}`,
    ];
    if (Array.isArray(payload.files)) {
      for (const file of payload.files) {
        const knowledgeSuffix = file.knowledge_fact_count ? ` | ${formatNumber(file.knowledge_fact_count)} fact(s)` : '';
        lines.push(`${getRelativePath(file.file_path)} :: ${formatNumber(file.words_processed)} words across ${formatNumber(file.chunks)} chunk(s)${knowledgeSuffix}`);
      }
    }
    state.trainingResult = lines.join('\n');
    await refreshRuntime();
    renderTraining();
    setAlert('Training run completed.', 'success');
  } catch (error) {
    state.trainingResult = `Training failed: ${error.message}`;
    renderTraining();
    setAlert(`Training failed: ${error.message}`, 'error');
  } finally {
    setBusy(elements.runTrainingButton, false);
  }
}

async function applySettings() {
  setBusy(elements.applySettingsButton, true, 'Applying');
  setAlert('Applying runtime settings...', 'info');
  try {
    const requests = [
      {
        id: 'profile',
        method: 'apply_performance_profile',
        params: { profile_name: elements.performanceProfile.value },
      },
      {
        id: 'enhanced',
        method: 'update_setting',
        params: { key: 'enhanced_intelligence', value: elements.enhancedIntelligence.checked },
      },
      {
        id: 'reasoning',
        method: 'update_setting',
        params: { key: 'advanced_reasoning', value: elements.advancedReasoning.checked },
      },
      {
        id: 'learning',
        method: 'update_setting',
        params: { key: 'adaptive_learning', value: elements.adaptiveLearning.checked },
      },
      {
        id: 'hardwareAdaptive',
        method: 'update_setting',
        params: { key: 'hardware_adaptive_mode', value: elements.hardwareAdaptiveMode.checked },
      },
      {
        id: 'creativity',
        method: 'update_setting',
        params: { key: 'creativity_factor', value: Number(elements.creativityFactor.value || 0.49) },
      },
      {
        id: 'quality',
        method: 'update_setting',
        params: { key: 'quality_threshold', value: Number(elements.qualityThreshold.value || 0.41) },
      },
      {
        id: 'responseLength',
        method: 'update_setting',
        params: { key: 'max_response_length', value: Number(elements.maxResponseLength.value || 25) },
      },
      {
        id: 'save',
        method: 'save_settings',
        params: {},
      },
    ];
    const batchPayload = await window.aiAstraBridge.batch(requests);
    const failures = (batchPayload.results || []).filter((item) => item.ok === false);
    if (failures.length) {
      throw new Error(failures.map((item) => item.error).join(' | '));
    }
    await refreshRuntime();
    setAlert('Settings updated.', 'success');
  } catch (error) {
    setAlert(`Settings update failed: ${error.message}`, 'error');
  } finally {
    setBusy(elements.applySettingsButton, false);
  }
}

async function resetSettings() {
  setBusy(elements.resetSettingsButton, true, 'Resetting');
  try {
    unwrapEnvelope(await window.aiAstraBridge.invoke('reset_settings', {}));
    await refreshRuntime();
    setAlert('Settings restored to defaults.', 'success');
  } catch (error) {
    setAlert(`Reset failed: ${error.message}`, 'error');
  } finally {
    setBusy(elements.resetSettingsButton, false);
  }
}

async function runMaintenance(methodName, button, message) {
  setBusy(button, true, 'Running');
  try {
    const payload = unwrapEnvelope(await window.aiAstraBridge.invoke(methodName, {}));
    await refreshRuntime();
    setAlert(payload.message || message, 'success');
  } catch (error) {
    setAlert(`${message} failed: ${error.message}`, 'error');
  } finally {
    setBusy(button, false);
  }
}

async function openFromDocs(key) {
  const target = state.service && state.service[key];
  if (!target) {
    setAlert('This shortcut is not available yet.', 'error');
    return;
  }
  const result = await window.aiAstraBridge.openPath(target);
  if (!result.ok) {
    setAlert(`Could not open path: ${result.error}`, 'error');
  }
}

function subscribeToBridge() {
  window.aiAstraBridge.onBackendLog((entry) => {
    state.logs.push(entry);
    if (state.logs.length > 400) {
      state.logs.shift();
    }
    renderLogs();
  });

  window.aiAstraBridge.onBackendState((payload) => {
    state.backendState = payload.state || state.backendState;
    if (state.backendState === 'stopped') {
      setAlert('Backend stopped. Restart the app to reconnect.', 'error');
    }
    renderChrome();
  });
}

function captureElements() {
  Object.assign(elements, {
    alertBar: document.getElementById('alertBar'),
    transportState: document.getElementById('transportState'),
    sessionId: document.getElementById('sessionId'),
    serviceUrl: document.getElementById('serviceUrl'),
    qualityGauge: document.getElementById('qualityGauge'),
    qualityGaugeValue: document.getElementById('qualityGaugeValue'),
    qualityGaugeCaption: document.getElementById('qualityGaugeCaption'),
    memoryBar: document.getElementById('memoryBar'),
    memoryValue: document.getElementById('memoryValue'),
    chainCount: document.getElementById('chainCount'),
    associationCount: document.getElementById('associationCount'),
    clusterCount: document.getElementById('clusterCount'),
    knowledgeFactCount: document.getElementById('knowledgeFactCount'),
    storageMode: document.getElementById('storageMode'),
    featureCloud: document.getElementById('featureCloud'),
    transcript: document.getElementById('transcript'),
    provenanceGoal: document.getElementById('provenanceGoal'),
    provenanceAction: document.getElementById('provenanceAction'),
    provenanceDirective: document.getElementById('provenanceDirective'),
    provenanceMode: document.getElementById('provenanceMode'),
    provenanceHygiene: document.getElementById('provenanceHygiene'),
    provenanceEvidence: document.getElementById('provenanceEvidence'),
    provenanceContinuity: document.getElementById('provenanceContinuity'),
    provenanceEpisode: document.getElementById('provenanceEpisode'),
    composerForm: document.getElementById('composerForm'),
    promptInput: document.getElementById('promptInput'),
    responseStats: document.getElementById('responseStats'),
    sendButton: document.getElementById('sendButton'),
    rewardButton: document.getElementById('rewardButton'),
    penalizeButton: document.getElementById('penalizeButton'),
    exportConversationButton: document.getElementById('exportConversationButton'),
    refreshSnapshotButton: document.getElementById('refreshSnapshotButton'),
    trainingExtensions: document.getElementById('trainingExtensions'),
    trainingPathList: document.getElementById('trainingPathList'),
    chunkSizeOverride: document.getElementById('chunkSizeOverride'),
    priorityBoost: document.getElementById('priorityBoost'),
    pickTrainingButton: document.getElementById('pickTrainingButton'),
    runTrainingButton: document.getElementById('runTrainingButton'),
    refreshTrainingButton: document.getElementById('refreshTrainingButton'),
    trainingPlanList: document.getElementById('trainingPlanList'),
    trainingResult: document.getElementById('trainingResult'),
    performanceProfile: document.getElementById('performanceProfile'),
    maxResponseLength: document.getElementById('maxResponseLength'),
    creativityFactor: document.getElementById('creativityFactor'),
    qualityThreshold: document.getElementById('qualityThreshold'),
    enhancedIntelligence: document.getElementById('enhancedIntelligence'),
    advancedReasoning: document.getElementById('advancedReasoning'),
    adaptiveLearning: document.getElementById('adaptiveLearning'),
    hardwareAdaptiveMode: document.getElementById('hardwareAdaptiveMode'),
    applySettingsButton: document.getElementById('applySettingsButton'),
    resetSettingsButton: document.getElementById('resetSettingsButton'),
    syncButton: document.getElementById('syncButton'),
    replayButton: document.getElementById('replayButton'),
    gcButton: document.getElementById('gcButton'),
    generationStats: document.getElementById('generationStats'),
    conversationCount: document.getElementById('conversationCount'),
    availableMemory: document.getElementById('availableMemory'),
    controlMethodCount: document.getElementById('controlMethodCount'),
    knowledgeConceptCount: document.getElementById('knowledgeConceptCount'),
    knowledgeEvidenceCount: document.getElementById('knowledgeEvidenceCount'),
    learningHealthMode: document.getElementById('learningHealthMode'),
    learningHealthAcceptance: document.getElementById('learningHealthAcceptance'),
    learningHealthTrend: document.getElementById('learningHealthTrend'),
    learningHealthGate: document.getElementById('learningHealthGate'),
    hardwareTier: document.getElementById('hardwareTier'),
    hardwareAccelerator: document.getElementById('hardwareAccelerator'),
    hardwarePreferredGpu: document.getElementById('hardwarePreferredGpu'),
    hardwareWorkers: document.getElementById('hardwareWorkers'),
    vocabIdMode: document.getElementById('vocabIdMode'),
    vocabLexiconSize: document.getElementById('vocabLexiconSize'),
    vocabIdChainCount: document.getElementById('vocabIdChainCount'),
    vocabTextChainCount: document.getElementById('vocabTextChainCount'),
    workflowList: document.getElementById('workflowList'),
    logFeed: document.getElementById('logFeed'),
    railState: document.getElementById('railState'),
    railBackend: document.getElementById('railBackend'),
    methodCount: document.getElementById('methodCount'),
    clearLogButton: document.getElementById('clearLogButton'),
  });
}

function wireEvents() {
  document.querySelectorAll('[data-tab]').forEach((button) => {
    button.addEventListener('click', () => activateTab(button.dataset.tab));
  });
  elements.composerForm.addEventListener('submit', handleSend);
  elements.rewardButton.addEventListener('click', () => applyFeedback(true));
  elements.penalizeButton.addEventListener('click', () => applyFeedback(false));
  elements.exportConversationButton.addEventListener('click', exportConversation);
  elements.refreshSnapshotButton.addEventListener('click', async () => {
    setBusy(elements.refreshSnapshotButton, true, 'Refreshing');
    try {
      await refreshRuntime();
      setAlert('Runtime snapshot refreshed.', 'success');
    } catch (error) {
      setAlert(`Refresh failed: ${error.message}`, 'error');
    } finally {
      setBusy(elements.refreshSnapshotButton, false);
    }
  });
  elements.pickTrainingButton.addEventListener('click', pickTrainingSources);
  elements.refreshTrainingButton.addEventListener('click', recalcTrainingPlans);
  elements.runTrainingButton.addEventListener('click', runTraining);
  elements.applySettingsButton.addEventListener('click', applySettings);
  elements.resetSettingsButton.addEventListener('click', resetSettings);
  elements.syncButton.addEventListener('click', () => runMaintenance('sync_brain_state', elements.syncButton, 'Brain state synchronized.'));
  elements.replayButton.addEventListener('click', () => runMaintenance('perform_memory_replay', elements.replayButton, 'Memory replay complete.'));
  elements.gcButton.addEventListener('click', () => runMaintenance('force_garbage_collection', elements.gcButton, 'Garbage collection complete.'));
  elements.clearLogButton.addEventListener('click', () => {
    state.logs = [];
    renderLogs();
  });
  document.querySelectorAll('[data-open-key]').forEach((button) => {
    button.addEventListener('click', () => openFromDocs(button.dataset.openKey));
  });
}

window.addEventListener('DOMContentLoaded', async () => {
  captureElements();
  wireEvents();
  subscribeToBridge();
  activateTab('command');
  setAlert('Starting the standalone app and backend service...', 'info');
  await bootstrap();
});
