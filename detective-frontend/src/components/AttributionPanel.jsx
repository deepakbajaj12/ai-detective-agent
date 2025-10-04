import React from 'react';
import Drawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';

// Highlight tokens inside clue text by wrapping matches
function highlightClue(text, tokens){
  if(!tokens || tokens.length === 0) return text;
  // Sort tokens by length desc to avoid partial overlap issues
  const ordered = [...tokens].sort((a,b)=> b.token.length - a.token.length);
  let html = text;
  ordered.forEach(t => {
    const escaped = t.token.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
    const re = new RegExp(`\\b${escaped}\\b`, 'gi');
    html = html.replace(re, (m)=>`<mark data-weight="${t.weight.toFixed(2)}">${m}</mark>`);
  });
  return html;
}

export default function AttributionPanel({ open, onClose, data }){
  // data: { attribution: [ { clue_id, clue_text, tokens:[{token, weight}] } ] }
  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx:{ width:{ xs: '100%', sm: 420 }, p:2 } }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb:1 }}>
        <Typography variant="h6">Attribution</Typography>
        <IconButton size="small" onClick={onClose}><CloseIcon fontSize="small" /></IconButton>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb:2 }}>
        Token importance extracted via transformer attention (fallback heuristic). Darker marks = higher weight.
      </Typography>
      <Stack spacing={2} sx={{ pr:1 }}>
        {data?.attribution?.map(block => {
          const maxW = Math.max(0, ...block.tokens.map(t=> t.weight));
          return (
            <Box key={block.clue_id} sx={{ border:'1px solid rgba(191,164,111,0.35)', p:1.2, borderRadius:1 }}>
              <Typography variant="caption" color="text.secondary">Clue #{block.clue_id}</Typography>
              <Box
                className="clue-text"
                sx={{ mt:0.5, fontSize:'.85rem', lineHeight:1.35, '& mark': { background:'none', px:0.3, borderRadius:0.5, color:'inherit', fontWeight:600 }, '& mark[data-weight]': { position:'relative' } }}
                dangerouslySetInnerHTML={{ __html: highlightClue(block.clue_text, block.tokens) }}
              />
              <Stack spacing={0.5} sx={{ mt:1 }}>
                {block.tokens.slice(0,5).map(t => (
                  <Stack key={t.token} direction="row" spacing={1} alignItems="center">
                    <Typography variant="caption" sx={{ width:80, fontFamily:'monospace' }}>{t.token}</Typography>
                    <LinearProgress variant="determinate" value={maxW ? (t.weight / maxW)*100 : 0} sx={{ flexGrow:1, height:6, borderRadius:3 }} />
                    <Typography variant="caption" sx={{ width:36, textAlign:'right' }}>{t.weight.toFixed(2)}</Typography>
                  </Stack>
                ))}
              </Stack>
            </Box>
          );
        })}
        {(!data || !data.attribution || data.attribution.length===0) && <Typography variant="body2" color="text.secondary">No attribution data. Trigger with the toggle in list.</Typography>}
      </Stack>
    </Drawer>
  );
}
