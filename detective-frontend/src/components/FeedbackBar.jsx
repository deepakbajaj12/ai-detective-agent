import React, { useState } from 'react';
import Stack from '@mui/material/Stack';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import CheckIcon from '@mui/icons-material/CheckCircleOutline';
import CloseIcon from '@mui/icons-material/HighlightOff';
import HelpIcon from '@mui/icons-material/HelpOutline';
import CircularProgress from '@mui/material/CircularProgress';

export default function FeedbackBar({ suspect, rank }) {
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(null); // 'confirm' | 'reject' | 'uncertain'

  const send = (decision) => {
    setLoading(true);
    fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        suspect_id: suspect.id,
        decision,
        rank_at_feedback: rank,
        composite_score: suspect.composite_score,
        ml_score: suspect.score,
        evidence_score: suspect.evidence_score,
        offense_boost: suspect.offense_boost
      })
    }).then(r=>r.json()).then(d => {
      if(!d.error) setDone(decision);
    }).finally(()=> setLoading(false));
  };

  return (
    <Stack direction="row" spacing={1} alignItems="center" sx={{ mt:1 }}>
      {loading && <CircularProgress size={18} />}
      {!loading && (
        <>
          <Tooltip title="Confirm suspect relevance"><span><IconButton size="small" color={done==='confirm'?'success':'default'} onClick={()=>send('confirm')}><CheckIcon fontSize="small" /></IconButton></span></Tooltip>
          <Tooltip title="Reject (not relevant)"><span><IconButton size="small" color={done==='reject'?'error':'default'} onClick={()=>send('reject')}><CloseIcon fontSize="small" /></IconButton></span></Tooltip>
          <Tooltip title="Uncertain / ambiguous"><span><IconButton size="small" color={done==='uncertain'?'warning':'default'} onClick={()=>send('uncertain')}><HelpIcon fontSize="small" /></IconButton></span></Tooltip>
        </>
      )}
    </Stack>
  );
}