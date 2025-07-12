import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActionArea
} from '@mui/material';
import GavelIcon from '@mui/icons-material/Gavel';
import SearchIcon from '@mui/icons-material/Search';
import DescriptionIcon from '@mui/icons-material/Description';
import HistoryIcon from '@mui/icons-material/History';
import LockIcon from '@mui/icons-material/Lock';

const heroBg = 'https://images.unsplash.com/photo-1589829545856-d10d557cf95f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80';

// Keep only 5 most relevant features
const features = [
  {
    icon: <SearchIcon sx={{ fontSize: 36, color: '#0a2463' }} />,
    title: 'Semantic Legal Search',
    desc: 'Find relevant Indian legal information instantly using advanced semantic search powered by AI.',
    bg: '#e3eafc',
  },
  {
    icon: <GavelIcon sx={{ fontSize: 36, color: '#d4af37' }} />,
    title: 'Intelligent Legal Chatbot',
    desc: 'Ask legal questions and get instant, AI-powered answers tailored to Indian law.',
    bg: '#fffbe6',
  },
  {
    icon: <LockIcon sx={{ fontSize: 36, color: '#0a2463' }} />,
    title: 'Simple Captcha Verification',
    desc: 'No login or registration required. Just solve a simple captcha to start chatting.',
    bg: '#e3eafc',
  },
  {
    icon: <HistoryIcon sx={{ fontSize: 36, color: '#d4af37' }} />,
    title: 'Session-Based Chat History',
    desc: 'Your chat history is stored only in your browser for privacy and is cleared when you close the tab.',
    bg: '#fffbe6',
  },
  {
    icon: <LockIcon sx={{ fontSize: 36, color: '#0a2463' }} />,
    title: 'Privacy-Focused',
    desc: 'No user data or chat history is stored on any server. Everything stays on your device.',
    bg: '#e3eafc',
  },
];

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#121212', color: 'white', fontFamily: 'Segoe UI, Arial, sans-serif' }}>
      {/* Hero Section */}
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          minHeight: { xs: 400, md: 520 },
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: `linear-gradient(120deg, #0a2463 70%, #d4af37 100%)`,
          overflow: 'hidden',
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            zIndex: 1,
            background: `linear-gradient(90deg, #0a2463cc 60%, #d4af3780 100%)`,
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            zIndex: 0,
            backgroundImage: `url(${heroBg})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            opacity: 0.25,
          }}
        />
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 2, py: 8 }}>
          <Grid container spacing={6} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h2" sx={{ fontWeight: 700, mb: 3, color: '#d4af37', fontSize: { xs: '2.2rem', md: '3.2rem' } }}>
                न्यायदूत
              </Typography>
              <Typography variant="h5" sx={{ mb: 4, color: '#fff', fontWeight: 400, maxWidth: 500 }}>
                Your AI-powered Legal Assistant for Indian Law
              </Typography>
              <Button
                size="large"
                sx={{
                  background: '#d4af37',
                  color: '#0a2463',
                  fontWeight: 700,
                  fontSize: '1.2rem',
                  px: 5,
                  py: 2,
                  borderRadius: '10px',
                  boxShadow: '0 4px 24px rgba(212,175,55,0.18)',
                  textTransform: 'none',
                  '&:hover': {
                    background: '#c4a030',
                  },
                }}
                onClick={() => navigate('/Nyayadoot/captcha')}
              >
                Start with न्यायदूत
              </Button>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  width: '100%',
                  height: { xs: 260, md: 340 },
                  borderRadius: 6,
                  boxShadow: 6,
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    inset: 0,
                    width: '100%',
                    height: '100%',
                    background: 'linear-gradient(90deg, #0a2463cc 60%, #d4af3780 100%)',
                    zIndex: 1,
                  }}
                />
                <img
                  src={heroBg}
                  alt="Legal AI Platform Interface"
                  style={{ width: '100%', height: '100%', objectFit: 'cover', filter: 'brightness(0.9)' }}
                />
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 2,
                    textAlign: 'center',
                  }}
                >
                  <Typography variant="h6" sx={{ color: 'white', fontWeight: 700, mb: 1 }}>
                    Intelligent Legal Analysis
                  </Typography>
                  <Box sx={{ bgcolor: 'rgba(255,255,255,0.18)', px: 2, py: 1, borderRadius: 2, display: 'inline-block' }}>
                    <Typography variant="body2" sx={{ color: 'white' }}>
                      Powered by advanced AI algorithms
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Box sx={{ py: 10, bgcolor: '#1a1a1a' }}>
        <Container maxWidth="lg">
          <Typography variant="h4" align="center" sx={{ fontWeight: 700, mb: 2, color: '#d4af37' }}>
            Key Features
          </Typography>
          <Typography align="center" sx={{ mb: 6, color: '#ccc', maxWidth: 700, mx: 'auto' }}>
            न्यायदूत offers a focused set of tools to empower your legal research and practice.
          </Typography>
          <Grid container spacing={4} justifyContent="center">
            {features.map((feature, idx) => (
              <Grid
                item
                xs={12}
                sm={6}
                md={idx < 3 ? 4 : 6}
                key={idx}
                sx={{ display: 'flex', justifyContent: 'center' }}
              >
                <Card
                  sx={{
                    bgcolor: feature.bg,
                    borderRadius: 4,
                    boxShadow: 4,
                    transition: 'transform 0.3s, box-shadow 0.3s',
                    width: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    '&:hover': {
                      transform: 'translateY(-8px) scale(1.03)',
                      boxShadow: 8,
                    },
                  }}
                >
                  <CardActionArea sx={{ p: 3, minHeight: 220 }}>
                    <Box sx={{ mb: 2 }}>{feature.icon}</Box>
                    <CardContent>
                      <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: '#0a2463' }}>{feature.title}</Typography>
                      <Typography variant="body2" sx={{ color: '#333', fontWeight: 500 }}>{feature.desc}</Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* CTA Section */}
      <Box sx={{ py: 10, position: 'relative', bgcolor: '#0a2463', overflow: 'hidden' }}>
        <Box sx={{ position: 'absolute', top: -80, left: -80, width: 220, height: 220, bgcolor: '#d4af37', borderRadius: '50%', opacity: 0.18, zIndex: 0 }} />
        <Box sx={{ position: 'absolute', bottom: -120, right: -120, width: 320, height: 320, bgcolor: '#d4af37', borderRadius: '50%', opacity: 0.18, zIndex: 0 }} />
        <Container maxWidth="md" sx={{ position: 'relative', zIndex: 2 }}>
          <Typography variant="h4" align="center" sx={{ fontWeight: 700, mb: 3, color: 'white' }}>
            Ready to Transform Your Legal Research?
          </Typography>
          <Typography align="center" sx={{ mb: 5, color: '#f3f3f3', fontSize: '1.2rem' }}>
            Start using न्यायदूत today and experience the future of legal assistance.
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center' }}>
            <Button
              size="large"
              sx={{
                background: '#d4af37',
                color: '#0a2463',
                fontWeight: 700,
                fontSize: '1.15rem',
                px: 5,
                py: 2,
                borderRadius: '10px',
                boxShadow: '0 4px 24px rgba(212,175,55,0.18)',
                textTransform: 'none',
                '&:hover': {
                  background: '#c4a030',
                },
              }}
              onClick={() => navigate('/Nyayadoot/captcha')}
            >
              Start with न्यायदूत
            </Button>
          </Box>
        </Container>
      </Box>

      {/* Footer */}
      <Box sx={{ py: 4, bgcolor: '#121212', textAlign: 'center', color: '#d4af37', fontWeight: 500, fontSize: '1.1rem' }}>
        © 2025 न्यायदूत. All rights reserved.
      </Box>
    </Box>
  );
};

export default LandingPage; 