const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { BackendService } = require('./backend_service');

const APP_ROOT = __dirname;
const DEFAULT_MAI_ROOT = path.resolve(APP_ROOT, '..');
const MAI_ROOT = path.resolve(process.env.MAI_ROOT || DEFAULT_MAI_ROOT);
const WORKSPACE_ROOT = path.resolve(process.env.MAI_WORKSPACE_ROOT || path.resolve(MAI_ROOT, '..'));
const DOCS_ROOT = path.join(WORKSPACE_ROOT, 'docs');
const RENDERER_INDEX = path.join(APP_ROOT, 'renderer', 'index.html');
const PRELOAD_PATH = path.join(APP_ROOT, 'preload.js');
const WINDOW_ICON = path.join(MAI_ROOT, 'mailogo.png');
const SMOKE_MODE = process.argv.includes('--smoke-test') || process.env.MAI_FRONTEND_SMOKE === '1';
const SMOKE_SCENARIO = process.env.MAI_FRONTEND_SMOKE_SCENARIO || 'full';
const SMOKE_OUTPUT_FILE = process.env.MAI_SMOKE_OUTPUT_FILE || path.join(APP_ROOT, '.last_smoke_result.json');
const USER_DATA_ROOT = process.env.MAI_FRONTEND_USER_DATA || path.join(MAI_ROOT, '.standalone_user_data');
const EFFECTIVE_USER_DATA_ROOT = SMOKE_MODE
  ? path.join(os.tmpdir(), `mai-standalone-smoke-${process.pid}`)
  : USER_DATA_ROOT;
const SMOKE_STATE_ROOT = SMOKE_MODE
  ? path.join(os.tmpdir(), `mai-standalone-state-${process.pid}`)
  : null;
if (SMOKE_STATE_ROOT) {
  process.env.MAI_STATE_DIR = SMOKE_STATE_ROOT;
}
const SMOKE_TRAINING_FILE = path.join(os.tmpdir(), `mai-standalone-training-smoke-${process.pid}.md`);
const SMOKE_SCREENSHOT_DIR = process.env.MAI_SMOKE_SCREENSHOT_DIR
  ? path.resolve(process.env.MAI_SMOKE_SCREENSHOT_DIR)
  : null;
const WINDOW_WIDTH = Number(process.env.MAI_WINDOW_WIDTH || 1560);
const WINDOW_HEIGHT = Number(process.env.MAI_WINDOW_HEIGHT || 960);
const RENDERER_METHODS = new Set([
  'apply_feedback',
  'collect_training_files',
  'export_conversation_text',
  'force_garbage_collection',
  'generate_response',
  'perform_memory_replay',
  'reset_settings',
  'sync_brain_state',
  'train_files',
]);
const RENDERER_BATCH_METHODS = new Set([
  'apply_performance_profile',
  'get_feature_runtime_snapshot',
  'get_runtime_bootstrap_snapshot',
  'get_session_info',
  'get_settings_snapshot',
  'get_training_plan',
  'save_settings',
  'update_setting',
]);

let mainWindow = null;
let quitInProgress = false;
let smokeResultWritten = false;

async function captureSmokeScreenshot(name) {
  if (!SMOKE_MODE || !SMOKE_SCREENSHOT_DIR || !mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  try {
    fs.mkdirSync(SMOKE_SCREENSHOT_DIR, { recursive: true });
    await mainWindow.webContents.executeJavaScript('window.scrollTo(0, 0); new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));', true);
    const image = await mainWindow.webContents.capturePage();
    fs.writeFileSync(path.join(SMOKE_SCREENSHOT_DIR, `${name}.png`), image.toPNG());
  } catch (_error) {
    // Screenshot capture is diagnostic only and must never fail the smoke run.
  }
}

app.setPath('userData', EFFECTIVE_USER_DATA_ROOT);
app.setPath('sessionData', path.join(EFFECTIVE_USER_DATA_ROOT, 'session'));

if (SMOKE_MODE) {
  try {
    fs.writeFileSync(SMOKE_OUTPUT_FILE, `${JSON.stringify({
      timestamp: new Date().toISOString(),
      argv: process.argv.slice(1),
      ok: false,
      phase: 'booting',
    })}\n`, 'utf8');
  } catch (_error) {
    // If the boot marker cannot be written, the later smoke guards will still try again.
  }
}

const backend = new BackendService({
  pythonPath: process.env.MAI_BACKEND_PYTHON,
  workspaceRoot: WORKSPACE_ROOT,
  maiRoot: MAI_ROOT,
});

backend.on('log', (entry) => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('mai:backend-log', entry);
  }
});

backend.on('state', (state) => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('mai:backend-state', { state });
  }
});

function mapBatchResults(batchPayload) {
  const resultMap = {};
  const results = Array.isArray(batchPayload && batchPayload.results) ? batchPayload.results : [];
  for (const item of results) {
    resultMap[item.id] = item;
  }
  return resultMap;
}

function isSafeExternalUrl(value) {
  try {
    const parsed = new URL(String(value || ''));
    return parsed.protocol === 'https:' || parsed.protocol === 'http:';
  } catch (_error) {
    return false;
  }
}

function isPathInsideWorkspace(value) {
  const resolved = path.resolve(String(value || ''));
  const relative = path.relative(WORKSPACE_ROOT, resolved);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

async function buildBootstrapPayload() {
  await backend.start();
  const [health, sessionPayload, manifestPayload, batchPayload] = await Promise.all([
    backend.getJson('/health'),
    backend.getJson('/session'),
    backend.getJson('/manifest'),
    backend.batch([
      { id: 'bootstrap', method: 'get_runtime_bootstrap_snapshot', params: {} },
      { id: 'features', method: 'get_feature_runtime_snapshot', params: {} },
      { id: 'settings', method: 'get_settings_snapshot', params: {} },
      { id: 'session', method: 'get_session_info', params: {} },
    ], { timeoutMs: 300000 }),
  ]);
  const batchResults = mapBatchResults(batchPayload);
  return {
    health,
    session: (sessionPayload && sessionPayload.session) || (batchResults.session && batchResults.session.result) || null,
    manifest: (manifestPayload && manifestPayload.manifest) || manifestPayload,
    startup: {
      bootstrap: batchResults.bootstrap ? batchResults.bootstrap.result : null,
      features: batchResults.features ? batchResults.features.result : null,
      settings: batchResults.settings ? batchResults.settings.result : null,
    },
    logs: backend.getRecentLogs(),
    service: {
      ...backend.getDescriptor(),
      docs_root: DOCS_ROOT,
      workspace_root: WORKSPACE_ROOT,
      atlas_note: path.join(DOCS_ROOT, 'SGM Codebase Atlas.md'),
      api_note: path.join(DOCS_ROOT, 'SGM', '12 Backend API and Transport.md'),
      knowledge_note: path.join(DOCS_ROOT, 'SGM', '14 Concept and Relation Memory.md'),
      frontend_note: path.join(DOCS_ROOT, 'SGM', '13 Standalone Frontend.md'),
    },
  };
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: Math.max(1180, WINDOW_WIDTH),
    height: Math.max(760, WINDOW_HEIGHT),
    minWidth: 900,
    minHeight: 650,
    backgroundColor: '#0b1014',
    autoHideMenuBar: true,
    icon: WINDOW_ICON,
    show: !SMOKE_MODE,
    webPreferences: {
      preload: PRELOAD_PATH,
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  mainWindow.loadFile(RENDERER_INDEX);
  mainWindow.once('ready-to-show', () => {
    if (mainWindow && !SMOKE_MODE) {
      mainWindow.show();
    }
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternalUrl(url)) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (url !== mainWindow.webContents.getURL()) {
      event.preventDefault();
    }
  });
}

async function executeRendererSmoke() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    throw new Error('Smoke mode window was not created.');
  }
  const script = `
    (async () => {
      const smokeScenario = ${JSON.stringify(SMOKE_SCENARIO)};
      const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
      const waitFor = async (predicate, timeoutMs, label) => {
        const start = Date.now();
        while (Date.now() - start < timeoutMs) {
          const result = predicate();
          if (result) {
            return result;
          }
          await sleep(200);
        }
        throw new Error('Timed out waiting for ' + label);
      };

      await waitFor(() => {
        const transport = document.getElementById('transportState')?.textContent;
        const session = document.getElementById('sessionId')?.textContent;
        const service = document.getElementById('serviceUrl')?.textContent;
        return transport === 'ONLINE' && session && !session.endsWith('--') && service && !service.includes('offline') && !service.includes('Connecting');
      }, 45000, 'transport state');
      if (smokeScenario !== 'training') {
        document.querySelector('[data-tab="command"]').click();
        await window.maiBridge.smokeCapture('01-chat-ready');

        const prompt = document.getElementById('promptInput');
        const sendButton = document.getElementById('sendButton');
        prompt.value = 'Provide one sentence describing Mai from the transport smoke test.';
        prompt.dispatchEvent(new Event('input', { bubbles: true }));
        sendButton.click();

        const outcome = await waitFor(() => {
          const assistantCount = document.querySelectorAll('.message--assistant').length;
          if (assistantCount >= 1) {
            return { ok: true };
          }
          const systemMessage = Array.from(document.querySelectorAll('.message--system .message__body')).pop()?.textContent || '';
          const alertBar = document.getElementById('alertBar');
          const alertText = alertBar?.textContent || '';
          const alertClass = alertBar?.className || '';
          const lastLogs = Array.from(document.querySelectorAll('.log-entry__body'))
            .slice(-5)
            .map((node) => node.textContent || '')
            .filter(Boolean);
          if (systemMessage) {
            return {
              ok: false,
              error: 'System message: ' + systemMessage,
              alertText,
              alertClass,
              lastLogs,
            };
          }
          if (alertClass.includes('alert-bar--error') && !sendButton.disabled) {
            return {
              ok: false,
              error: 'Alert error: ' + alertText,
              alertText,
              alertClass,
              lastLogs,
            };
          }
          return null;
        }, 300000, 'assistant response');

        if (!outcome.ok) {
          throw new Error(JSON.stringify(outcome));
        }
        await window.maiBridge.smokeCapture('02-chat-response');
      }

      document.querySelector('[data-tab="training"]').click();
      await window.maiBridge.smokeCapture('03-training');
      const trainingSource = ${JSON.stringify(SMOKE_TRAINING_FILE)};
      state.selectedTrainingPaths = [trainingSource];
      renderTraining();
      await recalcTrainingPlans();

      const trainingPlanOutcome = await waitFor(() => {
        const planCards = document.querySelectorAll('#trainingPlanList .plan-card');
        const trainingResult = document.getElementById('trainingResult')?.textContent || '';
        const alertBar = document.getElementById('alertBar');
        const alertText = alertBar?.textContent || '';
        if (planCards.length >= 1) {
          return {
            ok: true,
            planCount: planCards.length,
            trainingResult,
          };
        }
        if (alertBar?.className.includes('alert-bar--error')) {
          return {
            ok: false,
            error: 'Training plan error: ' + alertText,
            trainingResult,
          };
        }
        return null;
      }, 120000, 'training plan');

      if (!trainingPlanOutcome.ok) {
        throw new Error(JSON.stringify(trainingPlanOutcome));
      }

      await runTraining();

      const trainingOutcome = await waitFor(() => {
        const trainingResult = document.getElementById('trainingResult')?.textContent || '';
        const alertBar = document.getElementById('alertBar');
        const alertText = alertBar?.textContent || '';
        const alertClass = alertBar?.className || '';
        if (trainingResult.includes('Successes:')) {
          return {
            ok: true,
            trainingResult,
            alertText,
          };
        }
        if (trainingResult.startsWith('Training failed:') || (alertClass.includes('alert-bar--error') && !document.getElementById('runTrainingButton')?.disabled)) {
          return {
            ok: false,
            error: 'Training run error: ' + (trainingResult || alertText),
            trainingResult,
            alertText,
          };
        }
        return null;
      }, 180000, 'training run');

      if (!trainingOutcome.ok) {
        throw new Error(JSON.stringify(trainingOutcome));
      }

      document.querySelector('[data-tab="systems"]').click();
      await window.maiBridge.smokeCapture('04-systems');
      const activeTab = document.querySelector('[data-tab="systems"]');
      const activePanel = document.querySelector('[data-panel="systems"]');
      if (activeTab?.getAttribute('aria-selected') !== 'true' || activePanel?.getAttribute('aria-hidden') !== 'false') {
        throw new Error('Tab accessibility state did not follow the active Systems view.');
      }
      const systemsOutcome = await waitFor(() => {
        const learningMode = document.getElementById('learningHealthMode')?.textContent || '';
        const hardwareTier = document.getElementById('hardwareTier')?.textContent || '';
        const hardwareAccelerator = document.getElementById('hardwareAccelerator')?.textContent || '';
        if (learningMode && hardwareTier && hardwareAccelerator) {
          return {
            learningMode,
            learningAcceptance: document.getElementById('learningHealthAcceptance')?.textContent || '',
            hardwareTier,
            hardwareAccelerator,
            hardwarePreferredGpu: document.getElementById('hardwarePreferredGpu')?.textContent || '',
          };
        }
        return null;
      }, 5000, 'systems metrics');

      return {
        ok: true,
        smokeScenario,
        transportState: document.getElementById('transportState')?.textContent || '',
        sessionLabel: document.getElementById('sessionId')?.textContent || '',
        tabCount: document.querySelectorAll('[data-tab]').length,
        assistantMessages: document.querySelectorAll('.message--assistant').length,
        latestAssistantText: Array.from(document.querySelectorAll('.message--assistant .message__body')).pop()?.textContent || '',
        generationStats: document.getElementById('generationStats')?.textContent || '',
        trainingPlanCount: trainingPlanOutcome.planCount || 0,
        trainingResult: trainingOutcome.trainingResult || '',
        learningMode: systemsOutcome.learningMode || '',
        learningAcceptance: systemsOutcome.learningAcceptance || '',
        hardwareTier: systemsOutcome.hardwareTier || '',
        hardwareAccelerator: systemsOutcome.hardwareAccelerator || '',
        hardwarePreferredGpu: systemsOutcome.hardwarePreferredGpu || '',
      };
    })();
  `;
  return mainWindow.webContents.executeJavaScript(script, true);
}

async function runSmokeMode() {
  const emitSmokeResult = (payload) => {
    if (smokeResultWritten) {
      return;
    }
    smokeResultWritten = true;
    const report = {
      timestamp: new Date().toISOString(),
      argv: process.argv.slice(1),
      ...payload,
    };
    const text = `${JSON.stringify(report)}\n`;
    fs.writeFileSync(SMOKE_OUTPUT_FILE, text, 'utf8');
    process.stdout.write(text);
  };
  try {
    fs.writeFileSync(SMOKE_TRAINING_FILE, [
      '# Mai Training Smoke',
      '',
      'This temporary file verifies that the standalone frontend training path is connected.',
      'The Electron renderer should calculate a plan and run a small training pass through the backend API.',
    ].join('\n'), 'utf8');
    await backend.start();
    const result = await executeRendererSmoke();
    emitSmokeResult(result);
    await safeQuitBackend();
    process.exit(0);
  } catch (error) {
    emitSmokeResult({ ok: false, error: error.message });
    await safeQuitBackend();
    process.exit(1);
  } finally {
    try {
      if (fs.existsSync(SMOKE_TRAINING_FILE)) {
        fs.unlinkSync(SMOKE_TRAINING_FILE);
      }
      if (SMOKE_STATE_ROOT && fs.existsSync(SMOKE_STATE_ROOT)) {
        fs.rmSync(SMOKE_STATE_ROOT, { recursive: true, force: true });
      }
    } catch (_error) {
      // Temporary smoke artifacts should not block shutdown.
    }
  }
}

function installSmokeGuards() {
  if (!SMOKE_MODE) {
    return;
  }

  const writeFailure = (error, source) => {
    if (smokeResultWritten) {
      return;
    }
    smokeResultWritten = true;
    const payload = {
      timestamp: new Date().toISOString(),
      argv: process.argv.slice(1),
      ok: false,
      source,
      error: error && error.message ? error.message : String(error),
    };
    fs.writeFileSync(SMOKE_OUTPUT_FILE, `${JSON.stringify(payload)}\n`, 'utf8');
  };

  process.on('uncaughtException', (error) => {
    writeFailure(error, 'uncaughtException');
  });
  process.on('unhandledRejection', (error) => {
    writeFailure(error, 'unhandledRejection');
  });
  app.on('render-process-gone', (_event, _webContents, details) => {
    writeFailure(new Error(`Renderer exited: ${details.reason}`), 'render-process-gone');
  });
}

async function safeQuitBackend() {
  try {
    await backend.shutdown();
  } catch (error) {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('mai:backend-log', {
        stream: 'stderr',
        text: `Standalone shutdown warning: ${error.message}`,
        timestamp: new Date().toISOString(),
      });
    }
  }
}

ipcMain.handle('mai:bootstrap', async () => buildBootstrapPayload());

ipcMain.handle('mai:invoke', async (_event, payload = {}) => {
  const method = String(payload.method || '');
  if (!RENDERER_METHODS.has(method)) {
    throw new Error(`Renderer method is not allowed: ${method}`);
  }
  return backend.invoke(method, payload.params || {});
});

ipcMain.handle('mai:batch', async (_event, requests = []) => {
  if (!Array.isArray(requests) || requests.length > 32) {
    throw new Error('Renderer batch must be an array with at most 32 items.');
  }
  for (const request of requests) {
    if (!request || !RENDERER_BATCH_METHODS.has(String(request.method || ''))) {
      throw new Error(`Renderer batch method is not allowed: ${String(request && request.method || '')}`);
    }
  }
  return backend.batch(requests);
});

ipcMain.handle('mai:pick-training-paths', async () => {
  const result = await dialog.showOpenDialog({
    title: 'Select Mai training sources',
    buttonLabel: 'Queue for Training',
    defaultPath: WORKSPACE_ROOT,
    properties: ['openFile', 'openDirectory', 'multiSelections', 'dontAddToRecent'],
    filters: [
      { name: 'Text and Markdown', extensions: ['md', 'txt', 'text', 'rst'] },
      { name: 'All files', extensions: ['*'] },
    ],
  });
  return result;
});

ipcMain.handle('mai:open-path', async (_event, targetPath) => {
  if (!targetPath) {
    return {
      ok: false,
      path: null,
      error: 'No path was provided.',
    };
  }
  const resolvedPath = path.resolve(String(targetPath || ''));
  if (!isPathInsideWorkspace(resolvedPath)) {
    return {
      ok: false,
      path: resolvedPath,
      error: 'The requested path is outside the Mai workspace.',
    };
  }
  if (!fs.existsSync(resolvedPath)) {
    return {
      ok: false,
      path: resolvedPath,
      error: 'The requested path does not exist.',
    };
  }
  const error = await shell.openPath(resolvedPath);
  return {
    ok: !error,
    path: resolvedPath,
    error: error || null,
  };
});

ipcMain.handle('mai:open-external', async (_event, url) => {
  if (!isSafeExternalUrl(url)) {
    return { ok: false, error: 'Only HTTP and HTTPS links may be opened.' };
  }
  await shell.openExternal(String(url));
  return { ok: true };
});

ipcMain.handle('mai:smoke-capture', async (_event, name) => {
  await captureSmokeScreenshot(String(name || 'screen').replace(/[^a-z0-9_-]/gi, '-'));
  return { ok: true };
});

app.whenReady().then(() => {
  installSmokeGuards();
  createWindow();
  if (SMOKE_MODE && mainWindow) {
    mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription) => {
      if (!smokeResultWritten) {
        const payload = {
          timestamp: new Date().toISOString(),
          argv: process.argv.slice(1),
          ok: false,
          source: 'did-fail-load',
          error: `${errorCode}: ${errorDescription}`,
        };
        smokeResultWritten = true;
        fs.writeFileSync(SMOKE_OUTPUT_FILE, `${JSON.stringify(payload)}\n`, 'utf8');
      }
    });
  }
  if (SMOKE_MODE) {
    mainWindow.webContents.once('did-finish-load', () => {
      runSmokeMode();
    });
  } else {
    backend.start().catch((error) => {
      if (SMOKE_MODE) {
        process.stdout.write(`${JSON.stringify({ ok: false, error: error.message })}\n`);
        app.exit(1);
        return;
      }
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('mai:backend-log', {
          stream: 'stderr',
          text: `Backend warm-up failed: ${error.message}`,
          timestamp: new Date().toISOString(),
        });
      }
    });
  }
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('before-quit', (event) => {
  if (quitInProgress) {
    return;
  }
  quitInProgress = true;
  event.preventDefault();
  safeQuitBackend().finally(() => {
    app.quit();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
