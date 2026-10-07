const { EventEmitter } = require('events');
const { spawn } = require('child_process');
const crypto = require('crypto');
const http = require('http');
const net = require('net');
const path = require('path');

const DEFAULT_HOST = '127.0.0.1';
const LOG_HISTORY_LIMIT = 400;
const REQUEST_TIMEOUT_MS = 120000;
const LONG_REQUEST_TIMEOUT_MS = 600000;
const STARTUP_TIMEOUT_MS = 30000;
const LONG_RUNNING_METHODS = new Set([
  'generate_response',
  'get_runtime_bootstrap_snapshot',
  'get_feature_runtime_snapshot',
  'train_files',
  'learn_from_training_chunk',
  'finalize_training_run',
  'import_brain_bundle',
  'import_hsb_data',
  'export_brain_bundle',
  'export_hsb_data',
]);

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stripOuterQuotes(value) {
  if (typeof value !== 'string') {
    return value;
  }
  if (value.startsWith('"') && value.endsWith('"') && value.length > 1) {
    return value.slice(1, -1);
  }
  return value;
}

function getTimeoutForMethod(methodName) {
  if (LONG_RUNNING_METHODS.has(String(methodName || ''))) {
    return LONG_REQUEST_TIMEOUT_MS;
  }
  return REQUEST_TIMEOUT_MS;
}

function getTimeoutForRequests(requests) {
  const items = Array.isArray(requests) ? requests : [];
  if (!items.length) {
    return REQUEST_TIMEOUT_MS;
  }
  return items.reduce((timeoutMs, request) => {
    return Math.max(timeoutMs, getTimeoutForMethod(request && request.method));
  }, REQUEST_TIMEOUT_MS);
}

function isPyLauncher(executable) {
  const baseName = path.basename(executable || '').toLowerCase();
  return baseName === 'py' || baseName === 'py.exe';
}

function getFreePort(host) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, host, () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      server.close((closeError) => {
        if (closeError) {
          reject(closeError);
          return;
        }
        resolve(port);
      });
    });
  });
}

class BackendService extends EventEmitter {
  constructor(options = {}) {
    super();
    this.host = options.host || DEFAULT_HOST;
    this.workspaceRoot = path.resolve(options.workspaceRoot || process.env.AI_ASTRA_WORKSPACE_ROOT || path.resolve(__dirname, '..', '..'));
    this.aiAstraRoot = path.resolve(options.aiAstraRoot || process.env.AI_ASTRA_ROOT || path.resolve(__dirname, '..'));
    this.stateDir = options.stateDir ? path.resolve(options.stateDir) : (process.env.AI_ASTRA_STATE_DIR || '');
    const defaultPython = process.platform === 'win32' ? 'py' : 'python3';
    this.pythonPath = stripOuterQuotes(options.pythonPath || process.env.AI_ASTRA_BACKEND_PYTHON || defaultPython);
    this.authToken = String(options.authToken || crypto.randomBytes(32).toString('hex'));
    this.requestTimeoutMs = Number(options.requestTimeoutMs || REQUEST_TIMEOUT_MS);
    this.startupTimeoutMs = Number(options.startupTimeoutMs || STARTUP_TIMEOUT_MS);
    this.process = null;
    this.port = 0;
    this.state = 'idle';
    this.logs = [];
    this.pendingBuffers = { stdout: '', stderr: '' };
    this.startPromise = null;
  }

  get baseUrl() {
    return this.port ? `http://${this.host}:${this.port}` : null;
  }

  getRecentLogs() {
    return [...this.logs];
  }

  getDescriptor() {
    return {
      state: this.state,
      host: this.host,
      port: this.port,
      base_url: this.baseUrl,
      workspace_root: this.workspaceRoot,
      ai_astra_root: this.aiAstraRoot,
      conversations_dir: path.join(this.aiAstraRoot, 'conversations'),
      docs_root: path.join(this.workspaceRoot, 'docs'),
      atlas_note: path.join(this.workspaceRoot, 'docs', 'SGM Codebase Atlas.md'),
      api_note: path.join(this.workspaceRoot, 'docs', 'SGM', '12 Backend API and Transport.md'),
      frontend_note: path.join(this.workspaceRoot, 'docs', 'SGM', '13 Standalone Frontend.md'),
      frontend_root: __dirname,
    };
  }

  _pushLog(stream, text) {
    const entry = {
      stream,
      text,
      timestamp: new Date().toISOString(),
    };
    this.logs.push(entry);
    if (this.logs.length > LOG_HISTORY_LIMIT) {
      this.logs.shift();
    }
    this.emit('log', entry);
  }

  _consumeLogChunk(stream, chunk) {
    this.pendingBuffers[stream] += chunk.toString('utf8');
    const lines = this.pendingBuffers[stream].split(/\r?\n/);
    this.pendingBuffers[stream] = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) {
        this._pushLog(stream, trimmed);
      }
    }
  }

  _flushLogBuffers() {
    for (const stream of ['stdout', 'stderr']) {
      const remainder = (this.pendingBuffers[stream] || '').trim();
      if (remainder) {
        this._pushLog(stream, remainder);
      }
      this.pendingBuffers[stream] = '';
    }
  }

  _buildSpawnConfig(port) {
    const args = [];
    if (isPyLauncher(this.pythonPath)) {
      args.push('-3');
    }
    args.push('-m', 'ai_astra_main.headless_api', 'serve-http', '--host', this.host, '--port', String(port));
    return {
      command: this.pythonPath,
      args,
    };
  }

  _markStopped(message, stream = 'stderr') {
    if (message) {
      this._pushLog(stream, message);
    }
    this.process = null;
    this.state = 'stopped';
    this.emit('state', this.state);
  }

  async start() {
    if (this.state === 'running' && this.process && !this.process.killed) {
      return this.getDescriptor();
    }
    if (this.startPromise) {
      return this.startPromise;
    }
    this.startPromise = this._startImpl().finally(() => {
      this.startPromise = null;
    });
    return this.startPromise;
  }

  async _startImpl() {
    this.port = await getFreePort(this.host);
    const spawnConfig = this._buildSpawnConfig(this.port);
    this.state = 'starting';
    this._pushLog('stdout', `Launching backend transport on ${this.host}:${this.port}`);

    const childEnv = {
      ...process.env,
      AI_ASTRA_BACKEND_AUTH_TOKEN: this.authToken,
      PYTHONIOENCODING: 'utf-8',
    };
    if (this.stateDir) {
      childEnv.AI_ASTRA_STATE_DIR = this.stateDir;
    }
    const child = spawn(spawnConfig.command, spawnConfig.args, {
      cwd: this.workspaceRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: childEnv,
      windowsHide: true,
    });

    this.process = child;
    child.stdout.on('data', (chunk) => this._consumeLogChunk('stdout', chunk));
    child.stderr.on('data', (chunk) => this._consumeLogChunk('stderr', chunk));
    const startupErrorPromise = new Promise((_, reject) => {
      child.once('error', (error) => {
        this._flushLogBuffers();
        if (this.process === child) {
          this._markStopped(`Backend transport failed to launch: ${error.message}`);
        } else {
          this._pushLog('stderr', `Backend transport failed to launch: ${error.message}`);
        }
        reject(new Error(`Could not launch backend transport: ${error.message}`));
      });
    });
    child.on('exit', (code, signal) => {
      this._flushLogBuffers();
      const message = `Backend transport exited (code=${code ?? 'null'}, signal=${signal ?? 'null'})`;
      if (this.process === child) {
        this._markStopped(message);
      } else {
        this._pushLog('stderr', message);
      }
    });

    this.emit('state', this.state);
    try {
      await Promise.race([this._waitForReady(), startupErrorPromise]);
      this.state = 'running';
      this.emit('state', this.state);
      return this.getDescriptor();
    } catch (error) {
      if (this.process === child && child.exitCode === null && !child.killed) {
        child.kill();
      }
      if (this.process === child) {
        this.process = null;
      }
      this.state = 'stopped';
      this.emit('state', this.state);
      throw error;
    }
  }

  async _waitForReady() {
    const startTime = Date.now();
    let lastError = null;

    while (Date.now() - startTime < this.startupTimeoutMs) {
      if (!this.process) {
        throw new Error('Backend process exited before it became ready.');
      }
      try {
        const response = await this.requestJson('GET', '/health');
        const healthPayload = response.payload;
        if (healthPayload && healthPayload.ok) {
          return healthPayload;
        }
        lastError = new Error('Backend health probe did not return ok.');
      } catch (error) {
        lastError = error;
      }
      await delay(250);
    }

    throw lastError || new Error('Timed out waiting for AI Astra backend transport to become ready.');
  }

  requestJson(method, routePath, payload, options = {}) {
    return new Promise((resolve, reject) => {
      const body = payload === undefined ? null : JSON.stringify(payload);
      const timeoutMs = Number(options.timeoutMs || this.requestTimeoutMs);
      const request = http.request(
        {
          host: this.host,
          port: this.port,
          path: routePath,
          method,
          timeout: timeoutMs,
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
            'Content-Length': body ? Buffer.byteLength(body) : 0,
            'X-AIAstra-Token': this.authToken,
          },
        },
        (response) => {
          let raw = '';
          response.setEncoding('utf8');
          response.on('data', (chunk) => {
            raw += chunk;
          });
          response.on('end', () => {
            try {
              const parsed = raw ? JSON.parse(raw) : {};
              resolve({
                statusCode: response.statusCode || 0,
                payload: parsed,
              });
            } catch (error) {
              reject(new Error(`Could not parse backend response from ${routePath}: ${error.message}`));
            }
          });
        },
      );

      request.on('timeout', () => {
        request.destroy(new Error(`Timed out calling ${routePath}`));
      });
      request.on('error', reject);
      if (body) {
        request.write(body);
      }
      request.end();
    });
  }

  async getJson(routePath, options = {}) {
    await this.start();
    const response = await this.requestJson('GET', routePath, undefined, options);
    return response.payload;
  }

  async postJson(routePath, payload, options = {}) {
    await this.start();
    const response = await this.requestJson('POST', routePath, payload, options);
    return response.payload;
  }

  async invoke(methodName, params = {}, options = {}) {
    return this.postJson('/api', {
      id: `${methodName}-${Date.now()}`,
      method: methodName,
      params,
    }, {
      timeoutMs: options.timeoutMs || getTimeoutForMethod(methodName),
    });
  }

  async batch(requests, options = {}) {
    return this.postJson('/api/batch', requests, {
      timeoutMs: options.timeoutMs || getTimeoutForRequests(requests),
    });
  }

  async shutdown(options = {}) {
    const forceAfterMs = Number(options.forceAfterMs || 2500);
    const child = this.process;
    if (!child) {
      this.state = 'stopped';
      return { ok: true, forced: false };
    }

    try {
      await this.postJson('/api', {
        id: 'electron-shutdown',
        method: 'shutdown',
        params: {},
      });
    } catch (error) {
      this._pushLog('stderr', `Shutdown request failed: ${error.message}`);
    }

    if (!this.process || child.exitCode !== null || child.killed) {
      this.state = 'stopped';
      return { ok: true, forced: false };
    }

    const exited = await new Promise((resolve) => {
      let finished = false;
      const cleanup = () => {
        if (finished) {
          return;
        }
        finished = true;
        resolve(true);
      };

      child.once('exit', cleanup);
      setTimeout(() => {
        if (!finished) {
          resolve(false);
        }
      }, forceAfterMs);
    });

    if (!exited && this.process) {
      this._pushLog('stderr', 'Backend did not stop cleanly in time. Forcing process termination.');
      this.process.kill();
    }

    this.state = 'stopped';
    return { ok: true, forced: !exited };
  }
}

module.exports = {
  BackendService,
};
