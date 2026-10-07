const { app, BrowserWindow, ipcMain } = require('electron');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { BackendService } = require('../standalone_frontend/backend_service');

const AI_ASTRA_ROOT = path.resolve(__dirname, '..');
const WORKSPACE_ROOT = path.resolve(AI_ASTRA_ROOT, '..');
const TEST_STATE_DIR = process.env.AI_ASTRA_TEST_USE_LIVE_STATE === '1'
  ? undefined
  : path.join(os.tmpdir(), `astra-test-state-${process.pid}`);
const backend = new BackendService({
  pythonPath: process.env.AI_ASTRA_BACKEND_PYTHON,
  workspaceRoot: WORKSPACE_ROOT,
  aiAstraRoot: AI_ASTRA_ROOT,
  stateDir: TEST_STATE_DIR,
});

const ALLOWED_METHODS = new Set([
  'generate_response',
  'get_runtime_bootstrap_snapshot',
  'get_feature_runtime_snapshot',
  'get_session_info',
]);

let windowRef = null;
let quitting = false;

backend.on('log', (entry) => {
  if (windowRef && !windowRef.isDestroyed()) {
    windowRef.webContents.send('astra-test:log', entry);
  }
});

backend.on('state', (state) => {
  if (windowRef && !windowRef.isDestroyed()) {
    windowRef.webContents.send('astra-test:state', { state });
  }
});

function createWindow() {
  windowRef = new BrowserWindow({
    width: 900,
    height: 760,
    minWidth: 640,
    minHeight: 560,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#101318',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });
  windowRef.loadFile(path.join(__dirname, 'index.html'));
  windowRef.once('ready-to-show', () => windowRef && windowRef.show());
  windowRef.on('closed', () => { windowRef = null; });
}

ipcMain.handle('astra-test:bootstrap', async () => {
  await backend.start();
  const [health, manifest, session] = await Promise.all([
    backend.getJson('/health'),
    backend.getJson('/manifest'),
    backend.getJson('/session'),
  ]);
  return {
    health,
    manifest,
    session,
    service: backend.getDescriptor(),
  };
});

ipcMain.handle('astra-test:invoke', async (_event, payload = {}) => {
  const method = String(payload.method || '');
  if (!ALLOWED_METHODS.has(method)) {
    throw new Error(`Testing client method is not allowed: ${method}`);
  }
  return backend.invoke(method, payload.params || {});
});

async function shutdown() {
  try {
    await backend.shutdown();
  } catch (_error) {
    // The test client should still close if the backend already exited.
  }
  if (TEST_STATE_DIR) {
    try { fs.rmSync(TEST_STATE_DIR, { recursive: true, force: true }); } catch (_error) { /* best effort */ }
  }
}

app.whenReady().then(() => {
  createWindow();
  if (process.platform === 'darwin') {
    app.on('activate', () => {
      if (!BrowserWindow.getAllWindows().length) createWindow();
    });
  }
});

app.on('before-quit', (event) => {
  if (quitting) return;
  event.preventDefault();
  quitting = true;
  shutdown().finally(() => app.quit());
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
