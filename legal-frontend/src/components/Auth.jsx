import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
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
  background: '#0F172A', // Dark navy
  surface: '#1E293B',    // Blue-gray
  primary: '#7C3AED',    // Violet
  secondary: '#94A3B8',  // Muted text
  text: '#F8FAFC',       // Light text
};

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(5),
  borderRadius: '16px',
  maxWidth: '400px',
  width: '100%',
  margin: theme.spacing(2),
  backgroundColor: COLORS.surface,
  color: COLORS.text,
  boxShadow: '0px 4px 20px rgba(124, 58, 237, 0.3)',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(2),
}));

const StyledButton = styled(Button)(() => ({
  padding: '12px',
  fontSize: '1rem',
  fontWeight: 600,
  borderRadius: '8px',
  textTransform: 'none',
  marginTop: '16px',
  backgroundColor: COLORS.primary,
  color: COLORS.text,
  '&:hover': {
    backgroundColor: '#8B5CF6', // Slightly lighter violet
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
      <StyledPaper elevation={3}>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h4" fontWeight="bold" color={COLORS.primary}>
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </Typography>
          <Typography variant="body2" color={COLORS.secondary} sx={{ mt: 1 }}>
            {isLogin ? 'Sign in to continue' : 'Join us today'}
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ borderRadius: '8px' }}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert severity="success" sx={{ borderRadius: '8px' }}>
            {success}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
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
                borderRadius: '8px',
                '& fieldset': {
                  borderColor: COLORS.primary,
                },
                '&:hover fieldset': {
                  borderColor: '#8B5CF6',
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
                borderRadius: '8px',
                '& fieldset': {
                  borderColor: COLORS.primary,
                },
                '&:hover fieldset': {
                  borderColor: '#8B5CF6',
                },
              },
            }}
            placeholder="Enter your password"
          />
          <StyledButton fullWidth type="submit">
            {isLogin ? 'Sign In' : 'Create Account'}
          </StyledButton>
        </form>

        <Box sx={{ textAlign: 'center' }}>
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
