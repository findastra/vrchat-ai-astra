const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('maiBridge', {
  bootstrap: () => ipcRenderer.invoke('mai:bootstrap'),
  invoke: (method, params = {}) => ipcRenderer.invoke('mai:invoke', { method, params }),
  batch: (requests) => ipcRenderer.invoke('mai:batch', requests),
  pickTrainingSources: () => ipcRenderer.invoke('mai:pick-training-paths'),
  openPath: (targetPath) => ipcRenderer.invoke('mai:open-path', targetPath),
  openExternal: (url) => ipcRenderer.invoke('mai:open-external', url),
  smokeCapture: (name) => ipcRenderer.invoke('mai:smoke-capture', name),
  onBackendLog: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('mai:backend-log', listener);
    return () => ipcRenderer.removeListener('mai:backend-log', listener);
  },
  onBackendState: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('mai:backend-state', listener);
    return () => ipcRenderer.removeListener('mai:backend-state', listener);
  },
});
