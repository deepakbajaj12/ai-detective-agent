const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

export async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const headers = new Headers(options.headers || {});
  const token = localStorage.getItem('auth_token');
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const finalOpts = { ...options, headers };
  const res = await fetch(url, finalOpts);
  const ct = res.headers.get('content-type') || '';
  // Parse body early for error clarity
  let bodyText = '';
  let bodyJson = null;
  try {
    if (ct.includes('application/json')) {
      bodyJson = await res.json();
      bodyText = JSON.stringify(bodyJson, null, 2);
    } else {
      bodyText = await res.text();
    }
  } catch (e) {
    // ignore parse errors
  }
  if (!res.ok) {
    // Specialized 401 handling
    if (res.status === 401) {
      const msg = bodyJson?.error === 'auth required' ? 'Authentication required – please login.' : (bodyJson?.error || 'Unauthorized');
      throw new Error(`HTTP 401: ${msg}`);
    }
    throw new Error(`HTTP ${res.status}: ${bodyText || res.statusText}`);
  }
  return bodyJson !== null ? bodyJson : bodyText;
}

export function getAuthToken() {
  return localStorage.getItem('auth_token');
}

export function isAuthenticated() {
  return !!getAuthToken();
}

export { API_BASE };
