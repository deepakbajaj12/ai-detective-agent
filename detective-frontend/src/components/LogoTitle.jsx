import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

/**
 * Sherlock / Victorian inspired layered 3D logo title.
 * Uses multiple absolutely positioned layers with text-shadows to simulate depth & emboss.
 * Accessible: main visible text rendered once; decorative layers use aria-hidden.
 */
export default function LogoTitle({ text = 'AI Detective', tagline }) {
  return (
    <Box sx={{ position: 'relative', display: 'inline-flex', flexDirection: 'column' }}>
      <Box className="sherlock-3d-wrapper">
        <span className="sherlock-3d-layer fill" aria-hidden>{text}</span>
        <span className="sherlock-3d-layer stroke" aria-hidden>{text}</span>
        <span className="sherlock-3d-layer shadow" aria-hidden>{text}</span>
        <Typography
          component="span"
          role="heading"
          aria-label={text}
          variant="h4"
          className="sherlock-3d-base"
        >
          {text}
        </Typography>
      </Box>
      {tagline && (
        <Typography variant="caption" sx={{ mt: 0.5, letterSpacing: '0.15em', fontFamily: 'Merriweather, serif', opacity: 0.85 }}>
          {tagline}
        </Typography>
      )}
    </Box>
  );
}
