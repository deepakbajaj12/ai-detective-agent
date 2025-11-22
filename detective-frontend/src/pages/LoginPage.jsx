import React, { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import { API_BASE } from '../apiBase';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('auth_token'));

  useEffect(()=> {
    setToken(localStorage.getItem('auth_token'));
  }, []);

  const submit = async () => {
    setError(null);
    if(!username || !password) { setError('Username & password required'); return; }
    setLoading(true);
    try {
      const body = JSON.stringify({ username, password });
      const res = await fetch(`${API_BASE}/api/auth/${mode}`, { method:'POST', headers:{ 'Content-Type':'application/json' }, body });
      const data = await res.json();
      if(!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      if(data.token) {
        localStorage.setItem('auth_token', data.token);
        setToken(data.token);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    setToken(null);
  };

  return (
    <Box>
      <Typography variant='h4' sx={{ mb:2, fontFamily:'Playfair Display, serif' }}>Authentication</Typography>
      <Typography variant='body2' sx={{ mb:3, maxWidth:620 }}>
        Register a user (first time) or login to obtain a bearer token. The token is stored in localStorage and automatically attached to protected API requests. Uploading documents and modifying suspects requires authentication.
      </Typography>
      <Card sx={{ maxWidth:480 }}>
        <CardContent>
          <Stack spacing={2}>
            {token && <Alert severity='success'>Logged in. Token stored.</Alert>}
            {error && <Alert severity='error'>{error}</Alert>}
            {!token && (
              <>
                <TextField label='Username' value={username} onChange={e=>setUsername(e.target.value)} autoComplete='username' />
                <TextField label='Password' type='password' value={password} onChange={e=>setPassword(e.target.value)} autoComplete='current-password' />
                <Stack direction='row' spacing={1}>
                  <Button variant={mode==='login' ? 'contained' : 'outlined'} disabled={loading} onClick={()=>setMode('login')}>Login</Button>
                  <Button variant={mode==='register' ? 'contained' : 'outlined'} disabled={loading} onClick={()=>setMode('register')}>Register</Button>
                  <Button variant='contained' disabled={loading} onClick={submit}>{mode==='login' ? 'Login' : 'Register'}</Button>
                </Stack>
              </>
            )}
            {token && <Button color='secondary' variant='outlined' onClick={logout}>Logout</Button>}
            {token && <Typography variant='caption' color='text.secondary'>Token: {token.slice(0,24)}…</Typography>}
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}
