import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Paper, Typography, Button, TextField } from '@mui/material';

function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export default function Captcha() {
  const [a] = useState(getRandomInt(1, 10));
  const [b] = useState(getRandomInt(1, 10));
  const [userAnswer, setUserAnswer] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (parseInt(userAnswer, 10) === a + b) {
      navigate('/Nyayadoot/chat');
    } else {
      setError('Incorrect answer. Please try again.');
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#0F172A', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Paper elevation={6} sx={{ p: 5, borderRadius: 4, bgcolor: '#1E293B', minWidth: 350, maxWidth: 400, mx: 2 }}>
        <Typography variant="h4" fontWeight={700} color="#d4af37" gutterBottom align="center">
          Captcha Verification
        </Typography>
        <Typography variant="body1" color="#F8FAFC" align="center" mb={3}>
          Please solve the captcha to continue to the chat.
        </Typography>
        <form onSubmit={handleSubmit}>
          <Box display="flex" alignItems="center" justifyContent="center" mb={2}>
            <Typography variant="h6" color="#F8FAFC" mr={2}>
              What is {a} + {b}?
            </Typography>
            <TextField
              id="captcha-question"
              type="number"
              value={userAnswer}
              onChange={e => setUserAnswer(e.target.value)}
              variant="outlined"
              size="small"
              sx={{ bgcolor: '#0F172A', borderRadius: 2, input: { color: '#F8FAFC' }, width: 90 }}
              required
            />
          </Box>
          {error && <Typography color="error" align="center" mb={2}>{error}</Typography>}
          <Button
            type="submit"
            variant="contained"
            fullWidth
            sx={{ bgcolor: '#d4af37', color: '#0a2463', fontWeight: 700, borderRadius: 2, py: 1.2, fontSize: '1.1rem', '&:hover': { bgcolor: '#c4a030' } }}
          >
            Verify
          </Button>
        </form>
      </Paper>
    </Box>
  );
} 