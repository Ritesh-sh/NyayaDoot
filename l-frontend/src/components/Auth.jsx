import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext.jsx';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';

// Color scheme consistent with Chat and ChatMessage
const COLORS = {
  background: '#121212', // Deep blue
  surface: '#1E293B',    // Blue-gray
  primary: '#d4af37',    // Gold
  secondary: '#94A3B8',  // Muted text
  text: '#F8FAFC',       // Light text
};

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(6),
  borderRadius: '20px',
  maxWidth: '440px',
  width: '100%',
  margin: theme.spacing(4, 'auto'),
  backgroundColor: COLORS.surface,
  color: COLORS.text,
  boxShadow: '0px 8px 32px rgba(212, 175, 55, 0.18)',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(3),
  alignItems: 'center',
}));

const StyledButton = styled(Button)(() => ({
  padding: '14px',
  fontSize: '1.1rem',
  fontWeight: 700,
  borderRadius: '10px',
  textTransform: 'none',
  marginTop: '20px',
  backgroundColor: COLORS.primary,
  color: '#0a2463',
  width: '100%',
  '&:hover': {
    backgroundColor: '#c4a030',
  },
}));

const ToggleButton = styled(Button)(() => ({
  textTransform: 'none',
  fontWeight: 500,
  color: COLORS.secondary,
  marginTop: '12px',
  '&:hover': {
    textDecoration: 'underline',
    backgroundColor: 'transparent',
  },
}));

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    try {
      const endpoint = isLogin ? '/api/login' : '/api/register';
      const response = await axios.post(`http://localhost:3001${endpoint}`, {
        email,
        password,
      });

      if (isLogin) {
        login(response.data.token);
        navigate('/chat');
      } else {
        setSuccess('Registration successful! Please login.');
        setIsLogin(true);
      }
    } catch (error) {
      setError(error.response?.data?.error || 'Something went wrong');
    }
  };

  return (
    <Container
      maxWidth={false}
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: COLORS.background,
      }}
    >
      <StyledPaper elevation={6}>
        <Box sx={{ textAlign: 'center', width: '100%' }}>
          <Typography variant="h4" fontWeight="bold" color={COLORS.primary} gutterBottom>
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </Typography>
          <Typography variant="body1" color={COLORS.secondary} sx={{ mb: 2 }}>
            {isLogin ? 'Sign in to continue to Nyayadoot' : 'Join Nyayadoot today'}
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ borderRadius: '8px', width: '100%' }}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert severity="success" sx={{ borderRadius: '8px', width: '100%' }}>
            {success}
          </Alert>
        )}

        <form onSubmit={handleSubmit} style={{ width: '100%' }}>
          <TextField
            fullWidth
            margin="normal"
            label="Email Address"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            variant="outlined"
            InputLabelProps={{ style: { color: COLORS.secondary } }}
            InputProps={{ style: { color: COLORS.text } }}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: '10px',
                '& fieldset': {
                  borderColor: COLORS.primary,
                },
                '&:hover fieldset': {
                  borderColor: '#c4a030',
                },
              },
            }}
            placeholder="Enter your email"
          />
          <TextField
            fullWidth
            margin="normal"
            label="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            variant="outlined"
            InputLabelProps={{ style: { color: COLORS.secondary } }}
            InputProps={{ style: { color: COLORS.text } }}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: '10px',
                '& fieldset': {
                  borderColor: COLORS.primary,
                },
                '&:hover fieldset': {
                  borderColor: '#c4a030',
                },
              },
            }}
            placeholder="Enter your password"
          />
          <StyledButton fullWidth type="submit">
            {isLogin ? 'Sign In' : 'Create Account'}
          </StyledButton>
        </form>

        <Box sx={{ textAlign: 'center', width: '100%' }}>
          <ToggleButton
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
              setSuccess('');
            }}
          >
            {isLogin ? 'Need an account? Register' : 'Already have an account? Login'}
          </ToggleButton>
        </Box>
      </StyledPaper>
    </Container>
  );
};

export default Auth;
