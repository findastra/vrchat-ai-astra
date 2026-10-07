const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('aiAstraBridge', {
  bootstrap: () => ipcRenderer.invoke('astra:bootstrap'),
  invoke: (method, params = {}) => ipcRenderer.invoke('astra:invoke', { method, params }),
  batch: (requests) => ipcRenderer.invoke('astra:batch', requests),
  pickTrainingSources: () => ipcRenderer.invoke('astra:pick-training-paths'),
  openPath: (targetPath) => ipcRenderer.invoke('astra:open-path', targetPath),
  openExternal: (url) => ipcRenderer.invoke('astra:open-external', url),
  smokeCapture: (name) => ipcRenderer.invoke('astra:smoke-capture', name),
  onBackendLog: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('astra:backend-log', listener);
    return () => ipcRenderer.removeListener('astra:backend-log', listener);
  },
  onBackendState: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('astra:backend-state', listener);
    return () => ipcRenderer.removeListener('astra:backend-state', listener);
  },
});
