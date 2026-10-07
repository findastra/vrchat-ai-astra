const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('maiTest', {
  bootstrap: () => ipcRenderer.invoke('mai-test:bootstrap'),
  invoke: (method, params = {}) => ipcRenderer.invoke('mai-test:invoke', { method, params }),
  onLog: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('mai-test:log', listener);
    return () => ipcRenderer.removeListener('mai-test:log', listener);
  },
  onState: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('mai-test:state', listener);
    return () => ipcRenderer.removeListener('mai-test:state', listener);
  },
});
