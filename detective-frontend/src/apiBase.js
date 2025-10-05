const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

export async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  try {
    const headers = new Headers(options.headers || {});
    // Inject bearer token if present
    const token = localStorage.getItem('auth_token');
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    const finalOpts = { ...options, headers };
    const res = await fetch(url, finalOpts);
    if (!res.ok) {
      const text = await res.text().catch(()=> '');
      throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
    }
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return await res.json();
    return await res.text();
  } catch (err) {
    console.error('apiFetch error', url, err);
    throw err;
  }
}

export { API_BASE };
