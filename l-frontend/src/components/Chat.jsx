import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import {
  Button, TextField, Container, Box, Typography, Link, IconButton,
  Drawer, List, ListItem, ListItemText, Divider, AppBar, Toolbar,
  CssBaseline, createTheme, ThemeProvider, Dialog, DialogTitle,
  DialogContent, DialogActions, Fade
} from '@mui/material';
import { Menu as MenuIcon, Logout, Send as SendIcon, Add as AddIcon } from '@mui/icons-material';
import ChatMessage from './ChatMessage';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function generateSessionId() {
  return Date.now().toString(36) + Math.random().toString(36).substring(2, 10);
}

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: { default: '#0F172A', paper: '#1E293B' },
    primary: { main: '#7C3AED', contrastText: '#ffffff' },
    secondary: { main: '#64748B' },
    text: { primary: '#F8FAFC', secondary: '#94A3B8' },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: { borderRadius: 12, textTransform: 'none', fontWeight: 600 },
      },
    },
  },
});

/* ── Typing dots animation ────────────────────────────────────────────── */
function TypingIndicator() {
  return (
    <Box sx={{ display: 'flex', gap: 0.6, alignItems: 'center', p: 1.5, pl: 2 }}>
      {[0, 1, 2].map((i) => (
        <Box
          key={i}
          sx={{
            width: 8, height: 8, borderRadius: '50%', bgcolor: '#d4af37',
            animation: 'bounce 1.4s infinite ease-in-out',
            animationDelay: `${i * 0.16}s`,
            '@keyframes bounce': {
              '0%, 80%, 100%': { transform: 'scale(0.6)', opacity: 0.4 },
              '40%': { transform: 'scale(1)', opacity: 1 },
            },
          }}
        />
      ))}
      <Typography variant="caption" sx={{ color: '#94A3B8', ml: 1, fontSize: '0.8rem' }}>
        NyayaDoot is thinking…
      </Typography>
    </Box>
  );
}

/* ═══════════════════════════════════════════════════════════════════════ */

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(() => generateSessionId());
  const [showPopup, setShowPopup] = useState(false);
  const [userMessageCount, setUserMessageCount] = useState(0);
  const POPUP_THRESHOLD = 5;
  const navigate = useNavigate();
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const suggestionPrompts = [
    "How can I register a private limited company in India?",
    "What are my rights if I get arrested?",
    "How do I file a consumer complaint?",
    "What are the legal steps for divorce in India?",
  ];

  const followUpChips = [
    "Show me relevant sections",
    "What are the related cases?",
    "What will happen if I don't act?",
    "Explain more",
  ];

  // Load history from sessionStorage once
  useEffect(() => {
    const stored = sessionStorage.getItem('chatHistory');
    if (stored) setHistory(JSON.parse(stored));
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);
  useEffect(scrollToBottom, [messages, loading, scrollToBottom]);

  /* ── Send a message (used for both typed input and chip clicks) ────── */
  const sendMessage = async (text) => {
    if (!text?.trim()) return;
    const userMsg = text.trim();
    const newMessages = [...messages, { type: 'user', content: userMsg }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    const count = userMessageCount + 1;
    setUserMessageCount(count);
    if (count % POPUP_THRESHOLD === 0) setShowPopup(true);

    try {
      const { data } = await axios.post(
        `${API_URL}/Nyayadoot/process-query`,
        { query: userMsg, session_id: currentSessionId },
        { timeout: 45000 }
      );
      const updated = [...newMessages, { type: 'bot', content: data.answer }];
      setMessages(updated);

      // Persist
      const session = {
        session_id: currentSessionId,
        messages: updated,
        preview: newMessages.find((m) => m.type === 'user')?.content?.substring(0, 40) || userMsg.substring(0, 40),
        last_message_time: new Date().toISOString(),
      };
      const updatedHistory = [session, ...history.filter((s) => s.session_id !== currentSessionId)];
      setHistory(updatedHistory);
      sessionStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
    } catch (error) {
      const errText =
        error.code === 'ECONNABORTED'
          ? 'Request timed out. The server might be busy.'
          : error.response?.data?.detail || 'Something went wrong. Please try again.';
      setMessages((prev) => [...prev, { type: 'bot', content: `⚠️ ${errText}`, isError: true }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleSend = () => sendMessage(input);

  const startNewChat = async () => {
    const newSid = generateSessionId();
    setMessages([]);
    setCurrentSessionId(newSid);
    setDrawerOpen(false);
    setUserMessageCount(0);
    try { await axios.post(`${API_URL}/Nyayadoot/reset-session?session_id=${newSid}`); } catch (_) {}
  };

  const handleHistoryClick = (session) => {
    setMessages(session.messages || []);
    setCurrentSessionId(session.session_id);
    setDrawerOpen(false);
  };

  /* ── Render ──────────────────────────────────────────────────────────── */
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {/* Popup Dialog */}
      <Dialog open={showPopup} onClose={() => setShowPopup(false)}>
        <DialogTitle>For Better Assistance</DialogTitle>
        <DialogContent>
          <Typography>
            For better assistance, please visit{' '}
            <Link href="http://82.112.226.107/vaadvivaad/" target="_blank" rel="noopener noreferrer"><b>Click Here</b></Link>.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPopup(false)} color="primary" variant="contained">Later</Button>
        </DialogActions>
      </Dialog>

      <Box sx={{ display: 'flex', height: '100vh', bgcolor: '#0F172A' }}>
        {/* ── Sidebar Drawer ── */}
        <Drawer anchor="left" open={drawerOpen} onClose={() => setDrawerOpen(false)}>
          <Box sx={{ width: 300, p: 2, bgcolor: '#1E293B', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" fontWeight="bold" sx={{ color: '#d4af37', mb: 1 }}>
              🗂️ Chat History
            </Typography>
            <Button
              fullWidth variant="contained" onClick={startNewChat} startIcon={<AddIcon />}
              sx={{ mb: 2, bgcolor: '#d4af37', color: '#0a2463', fontWeight: 700, '&:hover': { bgcolor: '#c4a030' } }}
            >
              New Chat
            </Button>
            <Divider sx={{ mb: 1, bgcolor: '#334155' }} />
            <List sx={{ flex: 1, overflowY: 'auto' }}>
              {history.length > 0 ? history.map((session, idx) => (
                <ListItem
                  button key={idx}
                  onClick={() => handleHistoryClick(session)}
                  selected={session.session_id === currentSessionId}
                  sx={{
                    borderRadius: 2, mb: 0.5,
                    bgcolor: session.session_id === currentSessionId ? 'rgba(212,175,55,0.15)' : 'transparent',
                    '&:hover': { bgcolor: 'rgba(212,175,55,0.08)' },
                  }}
                >
                  <ListItemText
                    primary={session.preview || session.messages?.[0]?.content?.substring(0, 30) || 'Chat'}
                    secondary={session.last_message_time ? new Date(session.last_message_time).toLocaleString() : ''}
                    primaryTypographyProps={{ noWrap: true, sx: { color: '#F8FAFC', fontWeight: session.session_id === currentSessionId ? 700 : 400 } }}
                    secondaryTypographyProps={{ sx: { color: '#64748B', fontSize: '0.75rem' } }}
                  />
                </ListItem>
              )) : (
                <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>No chats yet.</Typography>
              )}
            </List>
          </Box>
        </Drawer>

        {/* ── Main Chat Area ── */}
        <Container maxWidth={false}
          sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', py: 0, alignItems: 'center', maxWidth: '960px', margin: '0 auto' }}
        >
          {/* Header */}
          <AppBar position="static" color="transparent" elevation={0} sx={{ maxWidth: '900px', width: '100%', bgcolor: 'transparent' }}>
            <Toolbar sx={{ display: 'flex', justifyContent: 'space-between', px: 0 }}>
              <IconButton onClick={() => setDrawerOpen(true)} sx={{ color: '#d4af37' }}><MenuIcon /></IconButton>
              <Typography variant="h5" fontWeight="bold" sx={{ color: '#d4af37', letterSpacing: 1 }}>⚖️ न्यायदूत</Typography>
              <IconButton onClick={() => navigate('/nyayadoot/')} sx={{ color: '#d4af37' }}><Logout /></IconButton>
            </Toolbar>
          </AppBar>

          {/* Messages */}
          <Box sx={{
            flex: 1, p: 2, overflowY: 'auto', bgcolor: '#1E293B', borderRadius: 3, boxShadow: 3,
            maxWidth: '900px', width: '100%', mx: 'auto', position: 'relative',
            '&::-webkit-scrollbar': { width: 6 },
            '&::-webkit-scrollbar-thumb': { bgcolor: '#334155', borderRadius: 3 },
          }}>
            {/* Empty state */}
            {messages.length === 0 && !loading && (
              <Fade in timeout={500}>
                <Box sx={{
                  position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                  p: 3, bgcolor: '#232946', borderRadius: 2, boxShadow: 3,
                  maxWidth: '600px', width: '90%', textAlign: 'center',
                }}>
                  <Typography variant="subtitle1" sx={{ color: '#d4af37', fontWeight: 700, mb: 1 }}>
                    Get Started with These Questions
                  </Typography>
                  <Divider sx={{ mb: 2, bgcolor: '#d4af37', height: 2, borderRadius: 1 }} />
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                    {suggestionPrompts.map((prompt, i) => (
                      <Button
                        key={i} variant="outlined"
                        onClick={() => { setInput(prompt); inputRef.current?.focus(); }}
                        sx={{
                          bgcolor: '#0F172A', color: '#d4af37', fontWeight: 500,
                          py: 1, px: 2, maxWidth: '520px', width: '100%',
                          fontSize: '0.93rem', whiteSpace: 'normal', wordBreak: 'break-word',
                          height: 'auto', border: '1.5px solid #d4af37', borderRadius: '20px',
                          transition: 'all 0.25s ease',
                          '&:hover': { bgcolor: '#d4af37', color: '#0a2463', borderColor: '#c4a030', transform: 'translateY(-2px)' },
                        }}
                      >
                        {prompt}
                      </Button>
                    ))}
                  </Box>
                </Box>
              </Fade>
            )}

            {/* Chat bubbles */}
            {messages.map((msg, idx) => (
              <Fade in timeout={300} key={idx}>
                <Box display="flex" justifyContent={msg.type === 'user' ? 'flex-end' : 'flex-start'} mb={1}>
                  <ChatMessage type={msg.type} answer={msg.content} isError={msg.isError} />
                </Box>
              </Fade>
            ))}

            {/* Typing indicator */}
            {loading && <TypingIndicator />}

            {/* Follow-up chips */}
            {!loading && messages.length > 0 && messages[messages.length - 1]?.type === 'bot' && !messages[messages.length - 1]?.isError && (
              <Fade in timeout={400}>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1.5, ml: 0.5 }}>
                  {followUpChips.map((chip, i) => (
                    <Button
                      key={i} size="small" variant="outlined"
                      onClick={() => sendMessage(chip)}
                      sx={{
                        color: '#94A3B8', borderColor: '#334155', borderRadius: '16px',
                        fontSize: '0.78rem', px: 1.5, py: 0.3, textTransform: 'none',
                        '&:hover': { borderColor: '#d4af37', color: '#d4af37', bgcolor: 'rgba(212,175,55,0.06)' },
                      }}
                    >
                      {chip}
                    </Button>
                  ))}
                </Box>
              </Fade>
            )}

            <div ref={messagesEndRef} />
          </Box>

          {/* Input Area */}
          <Box sx={{
            p: 1.5, display: 'flex', gap: 1.5, mt: 1.5, mb: 1,
            bgcolor: '#1E293B', borderRadius: 2, boxShadow: 1,
            maxWidth: '900px', width: '100%', mx: 'auto',
          }}>
            <TextField
              fullWidth variant="outlined" value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
              disabled={loading} placeholder="Ask your legal question..."
              inputRef={inputRef} size="small"
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '10px', color: '#F8FAFC',
                  '& fieldset': { borderColor: '#334155' },
                  '&:hover fieldset': { borderColor: '#d4af37' },
                  '&.Mui-focused fieldset': { borderColor: '#d4af37' },
                },
              }}
            />
            <IconButton
              onClick={handleSend} disabled={loading || !input.trim()}
              sx={{
                bgcolor: '#d4af37', color: '#0a2463', width: 42, height: 42,
                '&:hover': { bgcolor: '#c4a030' },
                '&.Mui-disabled': { bgcolor: '#334155', color: '#64748B' },
              }}
            >
              <SendIcon fontSize="small" />
            </IconButton>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}
