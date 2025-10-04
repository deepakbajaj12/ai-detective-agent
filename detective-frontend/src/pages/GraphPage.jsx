import React, { useEffect, useState } from 'react';
import Typography from '@mui/material/Typography';
import LinearProgress from '@mui/material/LinearProgress';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import { apiFetch } from '../apiBase';

// Minimal inline D3-like force simulation (lightweight fallback)
function runLayout(nodes, edges, iterations=200){
  const pos = nodes.map((_,i)=> ({ x: Math.cos(i)*100, y: Math.sin(i)*100 }));
  const idIndex = Object.fromEntries(nodes.map((n,i)=> [n.id, i]));
  for(let it=0; it<iterations; it++){
    // repulsion
    for(let i=0;i<nodes.length;i++){
      for(let j=i+1;j<nodes.length;j++){
        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const dist2 = dx*dx+dy*dy+0.1;
        const force = 400 / dist2;
        const fx = force*dx; const fy = force*dy;
        pos[i].x += fx; pos[i].y += fy;
        pos[j].x -= fx; pos[j].y -= fy;
      }
    }
    // attraction
    edges.forEach(e => {
      const si = idIndex[e.source]; const ti = idIndex[e.target];
      if(si==null || ti==null) return;
      const dx = pos[ti].x - pos[si].x;
      const dy = pos[ti].y - pos[si].y;
      pos[si].x += dx*0.01; pos[si].y += dy*0.01;
      pos[ti].x -= dx*0.01; pos[ti].y -= dy*0.01;
    });
  }
  return pos.map((p,i)=> ({ ...nodes[i], x: p.x, y: p.y }));
}

export default function GraphPage(){
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(()=> {
    apiFetch('/api/graph')
      .then(setGraph)
      .catch(()=> setGraph({ error:'Failed'}))
      .finally(()=> setLoading(false));
  }, []);
  if(loading) return <LinearProgress />;
  if(!graph || graph.error) return <Typography color="error">Failed to load graph.</Typography>;
  const laidOut = runLayout(graph.nodes, graph.edges, 120);
  const minX = Math.min(...laidOut.map(n=>n.x));
  const minY = Math.min(...laidOut.map(n=>n.y));
  const maxX = Math.max(...laidOut.map(n=>n.x));
  const maxY = Math.max(...laidOut.map(n=>n.y));
  const w = maxX - minX + 80;
  const h = maxY - minY + 80;
  const colorFor = (t) => t==='suspect' ? '#d32f2f' : t==='offense' ? '#ed6c02' : '#1976d2';
  return (
    <Stack spacing={2}>
      <Typography variant="h4" sx={{ fontFamily:'Playfair Display, serif' }}>Relationship Graph</Typography>
      <Paper variant="outlined" sx={{ p:1, overflow:'auto' }}>
        <svg width={w} height={h} style={{ maxWidth:'100%' }}>
          {graph.edges.map((e,i)=> {
            const s = laidOut.find(n=> n.id===e.source);
            const t = laidOut.find(n=> n.id===e.target);
            if(!s||!t) return null;
            return <g key={i}>
              <line x1={s.x-minX+40} y1={s.y-minY+40} x2={t.x-minX+40} y2={t.y-minY+40} stroke="#999" strokeWidth={1} markerEnd="url(#arrow)" />
            </g>;
          })}
          <defs>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L6,3 z" fill="#999" />
            </marker>
          </defs>
          {laidOut.map(n => (
            <g key={n.id}>
              <circle cx={n.x-minX+40} cy={n.y-minY+40} r={14} fill={colorFor(n.type)} fillOpacity={0.85} />
              <text x={n.x-minX+40} y={n.y-minY+40+3} textAnchor="middle" fontSize={8} fill="#fff">{n.label.slice(0,10)}</text>
            </g>
          ))}
        </svg>
      </Paper>
    </Stack>
  );
}
