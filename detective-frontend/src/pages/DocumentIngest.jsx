import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import LinearProgress from '@mui/material/LinearProgress';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import UploadFileIcon from '@mui/icons-material/UploadFile';

export default function DocumentIngest() {
  const [file, setFile] = useState(null);
  const [autoClues, setAutoClues] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = () => {
    if(!file) return;
    setUploading(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append('file', file);
    fd.append('auto_clues', autoClues ? '1' : '0');
    fetch('/api/documents/upload', { method:'POST', body: fd })
      .then(r => r.json())
      .then(data => {
        if(data.error) setError(data.error); else setResult(data);
      })
      .catch(e => setError(e.message))
      .finally(()=> setUploading(false));
  };

  return (
    <Box>
      <Typography variant='h4' sx={{ mb:2, fontFamily:'Playfair Display, serif' }}>PDF Ingestion</Typography>
      <Typography variant='body2' sx={{ mb:3, maxWidth:760 }}>
        Upload an investigative PDF (transcript, report, notes). Text is extracted server-side, optionally split into clues, and a suspect suggestion list is generated via the ML model. This accelerates case bootstrapping.
      </Typography>
      <Stack spacing={2} sx={{ maxWidth:480 }}>
        <Button variant='outlined' component='label' startIcon={<UploadFileIcon />} disabled={uploading}>
          {file? `Selected: ${file.name}` : 'Choose PDF'}
          <input type='file' accept='application/pdf' hidden onChange={e=> setFile(e.target.files?.[0] || null)} />
        </Button>
        <FormControlLabel control={<Switch checked={autoClues} onChange={e=> setAutoClues(e.target.checked)} />} label='Auto-create clues from lines' />
        <Button variant='contained' onClick={handleUpload} disabled={!file || uploading}>Upload & Analyze</Button>
        {uploading && <LinearProgress />}
        {error && <Alert severity='error'>{error}</Alert>}
        {result && (
          <Card>
            <CardContent>
              <Typography variant='h6' sx={{ mb:1 }}>Ingestion Result</Typography>
              <Typography variant='body2' sx={{ mb:1 }}>Chars extracted: {result.chars}</Typography>
              <Typography variant='subtitle2'>Suspect Suggestions:</Typography>
              <Stack direction='row' spacing={1} sx={{ flexWrap:'wrap', mt:1 }}>
                {(result.suspect_suggestions||[]).map(s => (
                  <Chip key={s.label} label={`${s.label} ${(s.score*100).toFixed(1)}%`} />
                ))}
                {(!result.suspect_suggestions || result.suspect_suggestions.length===0) && <Typography variant='caption' color='text.secondary'>None produced.</Typography>}
              </Stack>
              <Typography variant='caption' color='text.secondary' sx={{ display:'block', mt:1 }}>Model suggestions are a starting point. Validate before acting.</Typography>
            </CardContent>
          </Card>
        )}
      </Stack>
    </Box>
  );
}
