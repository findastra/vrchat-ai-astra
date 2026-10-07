const $ = (id) => document.getElementById(id);
const transport = $('transport');
const endpoint = $('endpoint');
const session = $('session');
const diagnostics = $('diagnostics');
const response = $('response');
const latency = $('latency');
const sendButton = $('sendButton');
const probeButton = $('probeButton');

function showDiagnostics(value) {
  diagnostics.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function setStatus(state) {
  transport.textContent = String(state || 'UNKNOWN').toUpperCase();
}

function unwrap(payload) {
  if (payload?.ok === false) throw new Error(payload.error || 'Backend request failed.');
  return Object.prototype.hasOwnProperty.call(payload || {}, 'result') ? payload.result : payload;
}

function resetControlTrace() {
  $('traceGoal').textContent = '--';
  $('traceAction').textContent = '--';
  $('traceDirective').textContent = '--';
  $('traceMode').textContent = '--';
  $('traceHygiene').textContent = '--';
  if ($('traceContinuity')) $('traceContinuity').textContent = '--';
  if ($('traceEpisode')) $('traceEpisode').textContent = '--';
}

function updateControlTrace(payload) {
  const metadata = payload?.metadata || {};
  const cognition = metadata.cognitive_trace || metadata.response_plan?.cognitive_control || {};
  const realization = metadata.realization_trace || {};
  const hygiene = metadata.learning_hygiene || {};
  const episode = cognition.episode_memory || {};
  $('traceGoal').textContent = cognition.goal || '--';
  $('traceAction').textContent = cognition.action || '--';
  $('traceDirective').textContent = cognition.directive || '--';
  const modeParts = [realization.mode || '--'];
  if (realization.preempted) modeParts.push('preempted');
  if (realization.cognitive_action) modeParts.push(realization.cognitive_action);
  $('traceMode').textContent = modeParts.join(' / ');
  if (typeof hygiene.accept === 'boolean') {
    const reason = hygiene.reason || (hygiene.accept ? 'accepted' : 'rejected');
    $('traceHygiene').textContent = `${hygiene.accept ? 'accept' : 'reject'} (${reason})`;
  } else {
    $('traceHygiene').textContent = '--';
  }
  if ($('traceContinuity')) {
    $('traceContinuity').textContent = cognition.goal_continuity ? 'continuing prior goal' : 'fresh selection';
  }
  if ($('traceEpisode')) {
    const preview = episode.bot_preview || episode.user_preview || '';
    $('traceEpisode').textContent = preview
      ? `score ${episode.score ?? '--'}: ${String(preview).slice(0, 100)}`
      : '--';
  }
}

async function probe() {
  probeButton.disabled = true;
  try {
    const payload = await window.maiTest.bootstrap();
    setStatus('online');
    endpoint.textContent = payload.service?.base_url || '--';
    session.textContent = payload.session?.session_id || '--';
    showDiagnostics(payload);
  } catch (error) {
    setStatus('error');
    showDiagnostics(`Startup failed: ${error.message}`);
  } finally {
    probeButton.disabled = false;
  }
}

async function send(event) {
  event.preventDefault();
  const userInput = $('prompt').value.trim();
  if (!userInput) return;
  sendButton.disabled = true;
  response.textContent = 'Waiting for backend response...';
  resetControlTrace();
  const started = performance.now();
  try {
    const payload = unwrap(await window.maiTest.invoke('generate_response', { user_input: userInput }));
    response.textContent = payload.response || '[empty response]';
    latency.textContent = `${Math.round(performance.now() - started)} ms | quality ${Math.round(Number(payload.quality_score || 0) * 100)}%`;
    updateControlTrace(payload);
    showDiagnostics(payload);
  } catch (error) {
    response.textContent = `Request failed: ${error.message}`;
    latency.textContent = `${Math.round(performance.now() - started)} ms | failed`;
    resetControlTrace();
    showDiagnostics(error.message);
  } finally {
    sendButton.disabled = false;
  }
}

$('chatForm').addEventListener('submit', send);
probeButton.addEventListener('click', probe);
$('clearButton').addEventListener('click', () => { diagnostics.textContent = ''; });
window.maiTest.onState(({ state }) => setStatus(state));
window.maiTest.onLog((entry) => {
  if (diagnostics.textContent === '') diagnostics.textContent = '';
  diagnostics.textContent += `${entry.stream}: ${entry.text}\n`;
});
probe();
