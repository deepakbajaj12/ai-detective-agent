import React from 'react';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';

export default function ChatPage(){
  const [q, setQ] = React.useState('');
  const [answer, setAnswer] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const ctrlRef = React.useRef(null);
  const send = async () => {
    if(!q.trim()) return;
    setAnswer('');
    setLoading(true);
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, case_id: 'default' }),
        signal: ctrl.signal,
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      while(true){
        const { done, value } = await reader.read();
        if(done) break;
        setAnswer(prev => prev + decoder.decode(value));
      }
    } catch(e) {
      setAnswer(prev => prev + '\n[stream aborted or failed]');
    } finally {
      setLoading(false);
    }
  };
  const stop = () => {
    if(ctrlRef.current){ ctrlRef.current.abort(); }
  };
  return (
    <Stack spacing={2}>
      <Typography variant="h4" sx={{ fontFamily:'Playfair Display, serif' }}>Chat</Typography>
      <Stack direction={{ xs:'column', sm:'row' }} spacing={2}>
        <TextField fullWidth label="Ask a question" value={q} onChange={e=>setQ(e.target.value)} />
        <Button variant="contained" onClick={send} disabled={loading}>Send</Button>
        <Button variant="outlined" onClick={stop} disabled={!loading}>Stop</Button>
      </Stack>
      <Paper variant="outlined" sx={{ p:2, minHeight: 160, whiteSpace:'pre-wrap' }}>{answer || 'Ask a question to begin.'}</Paper>
    </Stack>
  );
}
