import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import {
  Button, TextField, Container, Box, Typography, IconButton,
  Drawer, List, ListItem, ListItemText, Divider, AppBar, Toolbar,
  CssBaseline, createTheme, ThemeProvider, Chip, Tooltip
} from '@mui/material';
import { Menu as MenuIcon, Logout } from '@mui/icons-material';
import ChatMessage from './ChatMessage';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(Date.now().toString());
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const messagesEndRef = useRef(null);

  const suggestionPrompts = [
    "How can I register a private limited company in India?",
    "How can I protect my business with contracts?",
    "What are the tax implications for small businesses in India?",
    "What are the legal requirements for starting a business?"
  ];

  const theme = createTheme({
    palette: {
      mode: 'dark',
      background: {
        default: '#0F172A',
        paper: '#1E293B'
      },
      primary: {
        main: '#7C3AED',
        contrastText: '#ffffff'
      },
      secondary: {
        main: '#64748B'
      },
      text: {
        primary: '#F8FAFC',
        secondary: '#94A3B8'
      }
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            textTransform: 'none',
            fontWeight: 600,
          }
        }
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            input: {
              color: '#F8FAFC',
              backgroundColor: '#1E293B'
            }
          }
        }
      },
      MuiChip: {
        styleOverrides: {
          root: {
            transition: 'all 0.3s ease',
            borderRadius: 16,
            padding: '8px 12px',
            fontSize: '0.9rem',
            whiteSpace: 'normal',
            wordBreak: 'break-word',
            height: 'auto',
            minHeight: '32px',
            lineHeight: '1.4',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
            }
          }
        }
      }
    }
  });

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get('http://localhost:3001/api/chats', {
          headers: { Authorization: `Bearer ${user.token}` }
        });
        setHistory(response.data);
      } catch (error) {
        console.error('Error fetching history:', error);
      }
    };
    if (user) fetchHistory();
  }, [user]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    setMessages(prev => [...prev, { type: 'user', content: input }]);
    setLoading(true);

    try {
      const aiResponse = await axios.post(
        'http://localhost:8000/process-query',
        { query: input },
        { headers: { Authorization: `Bearer ${user.token}` } }
      );

      await axios.post(
        'http://localhost:3001/api/chats',
        {
          message: input,
          response: aiResponse.data.answer,
          session_id: currentSessionId
        },
        { headers: { Authorization: `Bearer ${user.token}` } }
      );

      setMessages(prev => [
        ...prev,
        { type: 'bot', content: aiResponse.data.answer }
      ]);

      const refreshHistory = await axios.get('http://localhost:3001/api/chats', {
        headers: { Authorization: `Bearer ${user.token}` }
      });
      setHistory(refreshHistory.data);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev,
        { type: 'bot', content: 'Sorry, an error occurred. Please try again.' }
      ]);
    } finally {
      setInput('');
      setLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setCurrentSessionId(Date.now().toString());
    setDrawerOpen(false);
  };

  const toggleDrawer = (open) => () => {
    setDrawerOpen(open);
  };

  const handleHistoryClick = (session) => {
    const sessionMessages = session.messages.flatMap(msg => [
      { type: 'user', content: msg.message },
      { type: 'bot', content: msg.response }
    ]);
    setMessages(sessionMessages);
    setCurrentSessionId(session.session_id);
    setDrawerOpen(false);
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'background.default' }}>

        <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer(false)}>
          <Box sx={{ width: 300, p: 2, bgcolor: 'background.paper', height: '100%' }}>
            <Typography variant="h6" fontWeight="bold" gutterBottom color="primary">
              🗂️ Chat History
            </Typography>
            <Button fullWidth variant="contained" onClick={startNewChat} sx={{ mb: 2 }}>
              New Chat
            </Button>
            <Divider sx={{ mb: 2 }} />
            <List>
              {history.length > 0 ? history.map((session, index) => (
                <ListItem
                  button
                  key={index}
                  onClick={() => handleHistoryClick(session)}
                  selected={session.session_id === currentSessionId}
                >
                  <ListItemText
                    primary={session.messages[0]?.message?.substring(0, 25) + "..."}
                    secondary={session.last_message_time ? new Date(session.last_message_time).toLocaleString() : 'No timestamp'}
                  />
                </ListItem>
              )) : (
                <Typography variant="body2" color="text.secondary">
                  No chats yet.
                </Typography>
              )}
            </List>
          </Box>
        </Drawer>

        <Container maxWidth="md" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', py: 2 }}>

          <AppBar position="static" color="transparent" elevation={0} sx={{ mb: 2 }}>
            <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <IconButton edge="start" onClick={toggleDrawer(true)} color="inherit">
                <MenuIcon />
              </IconButton>

              <Typography variant="h5" fontWeight="bold" color="primary">
                ⚖️ Nyayadoot
              </Typography>

              <IconButton onClick={handleLogout} color="inherit">
                <Logout />
              </IconButton>
            </Toolbar>
          </AppBar>

          <Box
            sx={{
              flex: 1,
              p: 2,
              overflowY: 'auto',
              bgcolor: 'background.paper',
              borderRadius: 3,
              boxShadow: 3,
              position: 'relative'
            }}
          >
            {messages.length === 0 && (
              <Box
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  p: 3,
                  bgcolor: 'background.paper',
                  borderRadius: 2,
                  boxShadow: 3,
                  maxWidth: '500px',
                  width: '100%',
                }}
              >
                <Typography
                  variant="subtitle1"
                  fontWeight="medium"
                  color="text.primary"
                  gutterBottom
                  align="center"
                >
                  Get Started with These Questions
                </Typography>
                <Box
                  sx={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'center',
                    gap: 1.5,
                    maxWidth: '100%',
                  }}
                >
                  {suggestionPrompts.map((prompt, index) => (
                    <Tooltip
                      key={index}
                      title={prompt}
                      arrow
                      placement="top"
                      enterDelay={300}
                      leaveDelay={100}
                      componentsProps={{
                        tooltip: {
                          sx: {
                            fontSize: '0.85rem',
                            maxWidth: 300,
                            whiteSpace: 'normal',
                            wordWrap: 'break-word',
                          }
                        }
                      }}
                    >
                      <Chip
                        label={
                          prompt.length > 90
                            ? prompt.substring(0, 37) + '...'
                            : prompt
                        }
                        onClick={() => setInput(prompt)}
                        sx={{
                          bgcolor: 'background.default',
                          color: 'text.primary',
                          fontWeight: 500,
                          py: 2.5,
                          px: 1.5,
                          minWidth: '120px',
                          maxWidth: '220px',
                          textAlign: 'center',
                          whiteSpace: 'normal',
                          wordBreak: 'break-word',
                          height: 'auto',
                          transition: 'all 0.3s ease',
                          borderRadius: '16px',
                          '&:hover': {
                            bgcolor: 'primary.main',
                            color: 'white',
                            transform: 'translateY(-2px)',
                            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
                          },
                        }}
                      />
                    </Tooltip>
                  ))}
                </Box>
              </Box>
            )}
            {messages.map((msg, index) => (
              <Box
                key={index}
                display="flex"
                justifyContent={msg.type === 'user' ? 'flex-end' : 'flex-start'}
                mb={1}
              >
                <ChatMessage type={msg.type} answer={msg.content} />
              </Box>
            ))}
            {loading && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Assistant is typing...
              </Typography>
            )}
            <div ref={messagesEndRef} />
          </Box>

          <Box
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
              borderTop: 1,
              borderColor: 'divider',
              mt: 2,
              bgcolor: 'background.paper',
              borderRadius: 2,
              boxShadow: 1
            }}
          >
            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                fullWidth
                variant="outlined"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                disabled={loading}
                placeholder="Ask your legal question..."
              />
              <Button
                variant="contained"
                onClick={handleSend}
                disabled={loading || !input.trim()}
                color="primary"
              >
                {loading ? 'Sending...' : 'Send'}
              </Button>
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}