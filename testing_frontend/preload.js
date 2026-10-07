const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('aiAstraTest', {
  bootstrap: () => ipcRenderer.invoke('astra-test:bootstrap'),
  invoke: (method, params = {}) => ipcRenderer.invoke('astra-test:invoke', { method, params }),
  onLog: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('astra-test:log', listener);
    return () => ipcRenderer.removeListener('astra-test:log', listener);
  },
  onState: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('astra-test:state', listener);
    return () => ipcRenderer.removeListener('astra-test:state', listener);
  },
});
