const fs = require('fs');
const path = require('path');

const APP_ROOT = __dirname;
const MAI_ROOT = path.resolve(APP_ROOT, '..');
const WORKSPACE_ROOT = path.resolve(MAI_ROOT, '..');

function candidatePaths() {
  const candidates = [];
  if (process.env.MAI_FRONTEND_ELECTRON) {
    candidates.push(process.env.MAI_FRONTEND_ELECTRON);
  }
  candidates.push(path.join(APP_ROOT, 'node_modules', 'electron', 'dist', 'electron.exe'));
  candidates.push(path.join(WORKSPACE_ROOT, 'maionline', 'node_modules', 'electron', 'dist', 'electron.exe'));
  return candidates;
}

function resolveElectronExecutable() {
  for (const candidate of candidatePaths()) {
    const resolved = path.resolve(String(candidate || ''));
    if (resolved && fs.existsSync(resolved)) {
      return resolved;
    }
  }
  return null;
}

module.exports = {
  APP_ROOT,
  MAI_ROOT,
  WORKSPACE_ROOT,
  candidatePaths,
  resolveElectronExecutable,
};
