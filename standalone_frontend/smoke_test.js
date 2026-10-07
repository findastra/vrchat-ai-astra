const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { BackendService } = require('./backend_service');
const { resolveElectronExecutable } = require('./electron_runtime');

async function runBackendServiceSmoke() {
  const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mai-backend-smoke-'));
  const backend = new BackendService({
    pythonPath: process.env.MAI_BACKEND_PYTHON,
    workspaceRoot: process.env.MAI_WORKSPACE_ROOT,
    stateDir,
  });

  backend.on('log', (entry) => {
    const stream = entry.stream === 'stderr' ? process.stderr : process.stdout;
    stream.write(`[${entry.stream}] ${entry.text}\n`);
  });

  try {
    await backend.start();
    const health = await backend.getJson('/health');
    const manifestPayload = await backend.getJson('/manifest');
    const batchPayload = await backend.batch([
      { id: 'session', method: 'get_session_info', params: {} },
      { id: 'status', method: 'get_runtime_bootstrap_snapshot', params: {} },
      { id: 'response', method: 'generate_response', params: { user_input: 'Summarize Mai in one sentence.' } },
    ]);

    if (!health || !health.ok) {
      throw new Error('Health probe did not return ok.');
    }
    if (!manifestPayload || !manifestPayload.manifest) {
      throw new Error('Manifest route did not return manifest data.');
    }
    if (!batchPayload || batchPayload.ok === false || !Array.isArray(batchPayload.results)) {
      throw new Error('Batch invocation failed.');
    }

    const resultMap = Object.fromEntries(batchPayload.results.map((item) => [item.id, item]));
    if (!resultMap.response || resultMap.response.ok === false) {
      throw new Error(`Generate request failed: ${(resultMap.response && resultMap.response.error) || 'unknown error'}`);
    }

    return {
      ok: true,
      base_url: backend.baseUrl,
      transport_methods: manifestPayload.manifest.transport.method_count,
      control_methods: manifestPayload.manifest.transport.control_methods.length,
      session_transport: resultMap.session.result.transport,
      response_quality: resultMap.response.result.quality_score,
      response_preview: String(resultMap.response.result.response || '').slice(0, 120),
    };
  } finally {
    await backend.shutdown({ forceAfterMs: 1500 });
    fs.rmSync(stateDir, { recursive: true, force: true });
  }
}

async function runInvalidPythonProbe() {
  const backend = new BackendService({
    pythonPath: 'L:/definitely-missing-python.exe',
    workspaceRoot: process.env.MAI_WORKSPACE_ROOT || process.cwd(),
    startupTimeoutMs: 1500,
    requestTimeoutMs: 300,
  });
  try {
    await backend.start();
    throw new Error('Invalid-python probe unexpectedly succeeded.');
  } catch (error) {
    return {
      ok: true,
      error_message: error.message,
    };
  } finally {
    await backend.shutdown({ forceAfterMs: 500 });
  }
}

function runChildProcess(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, options);
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', reject);
    child.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });
  });
}

async function runUiSmoke(scenario = 'full') {
  const outputPath = path.join(os.tmpdir(), `mai-standalone-smoke-${process.pid}-${Date.now()}.json`);
  const electronExecutable = resolveElectronExecutable();
  if (!electronExecutable) {
    throw new Error('Could not find an Electron runtime for the standalone frontend smoke test.');
  }
  if (fs.existsSync(outputPath)) {
    fs.unlinkSync(outputPath);
  }
  const childEnv = { ...process.env };
  delete childEnv.ELECTRON_RUN_AS_NODE;
  childEnv.MAI_FRONTEND_SMOKE = '1';
  childEnv.MAI_FRONTEND_SMOKE_SCENARIO = scenario;
  childEnv.MAI_SMOKE_OUTPUT_FILE = outputPath;
  const result = await runChildProcess(
    electronExecutable,
    [path.resolve(__dirname), '--smoke-test'],
    {
      cwd: path.resolve(__dirname),
      env: childEnv,
      windowsHide: true,
    },
  );
  const payloadText = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8') : '';
  if (fs.existsSync(outputPath)) {
    fs.unlinkSync(outputPath);
  }
  const lines = payloadText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const lastLine = lines[lines.length - 1] || '';
  let payload = null;
  try {
    payload = lastLine ? JSON.parse(lastLine) : null;
  } catch (error) {
    throw new Error(`Could not parse UI smoke output: ${error.message}`);
  }
  if (result.code !== 0 || !payload || payload.ok === false) {
    throw new Error((payload && payload.error) || result.stderr.trim() || `UI smoke exited with code ${result.code}`);
  }
  return {
    ...payload,
    electron_executable: electronExecutable,
    stderr_lines: result.stderr.split(/\r?\n/).map((line) => line.trim()).filter(Boolean),
  };
}

async function main(mode = (process.argv[2] || 'all')) {
  const outputPath = process.env.MAI_SMOKE_OUTPUT_FILE || '';

  if (mode === 'backend') {
    const payload = await runBackendServiceSmoke();
    if (outputPath) {
      fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
    }
    return payload;
  }

  if (mode === 'ui') {
    const payload = await runUiSmoke('full');
    if (outputPath) {
      fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
    }
    return payload;
  }

  if (mode === 'training') {
    const payload = await runUiSmoke('training');
    if (outputPath) {
      fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
    }
    return payload;
  }

  const payload = {
    backend: await runBackendServiceSmoke(),
    invalid_python_probe: await runInvalidPythonProbe(),
    ui: await runUiSmoke(),
  };
  if (outputPath) {
    fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  }
  return payload;
}

if (require.main === module) {
  main().then((payload) => {
    if (payload !== undefined) {
      console.log(JSON.stringify(payload, null, 2));
    }
  }).catch((error) => {
    console.error(error.stack || String(error));
    process.exitCode = 1;
  });
}

module.exports = {
  main,
  runBackendServiceSmoke,
  runInvalidPythonProbe,
  runUiSmoke,
};
