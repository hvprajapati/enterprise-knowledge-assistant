const BASE = 'http://127.0.0.1:8000/api/v1';

async function request(method, path, body = null, opts = {}) {
  const url = `${BASE}${path}`;
  const config = {
    method,
    headers: { ...opts.headers },
    ...opts,
  };

  if (body && !(body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json';
    config.body = JSON.stringify(body);
  } else if (body instanceof FormData) {
    config.body = body;
  }

  const res = await fetch(url, config);
  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const msg = data.detail || data.error || `HTTP ${res.status}`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }

  return data;
}

export const api = {
  health: () => fetch('http://127.0.0.1:8000/health').then(r => r.json()),

  query: (question) =>
    request('POST', '/query', { question }),

  streamQuery: async function* (question) {
    const res = await fetch(`${BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  },

  upload: async (file, onProgress) => {
    const fd = new FormData();
    fd.append('file', file);
    return request('POST', '/upload', fd, {});
  },

  listFiles: () => request('GET', '/uploads'),

  deleteFile: (filename) =>
    request('DELETE', `/uploads/${encodeURIComponent(filename)}`),

  indexDirectory: (directory) =>
    request('POST', '/index', { directory }),

  getJob: (jobId) => request('GET', `/jobs/${jobId}`),
};
