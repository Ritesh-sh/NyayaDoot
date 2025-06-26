import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { Container, List, ListItem, ListItemText, Typography, Box, Paper } from '@mui/material';

export default function History() {
  const [chats, setChats] = useState([]);
  const { user } = useAuth();

  useEffect(() => {
    const fetchChats = async () => {
      try {
        const response = await axios.get('http://localhost:3001/api/chats', {
          headers: { Authorization: `Bearer ${user.token}` }
        });
        setChats(response.data);
      } catch (error) {
        console.error('Error fetching history:', error);
      }
    };
    
    if (user) fetchChats();
  }, [user]);

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#121212', py: 6 }}>
      <Container maxWidth="md">
        <Typography variant="h4" gutterBottom sx={{ color: '#d4af37', fontWeight: 700, mb: 4, textAlign: 'center', letterSpacing: 1 }}>
          Chat History
        </Typography>
        <List sx={{ width: '100%', maxWidth: 800, mx: 'auto' }}>
          {chats.length === 0 ? (
            <Typography variant="body1" sx={{ color: '#fff', textAlign: 'center', mt: 6 }}>
              No chat history found.
            </Typography>
          ) : (
            chats.map((chat, index) => (
              <ListItem key={index} disablePadding sx={{ mb: 2 }}>
                <Paper
                  elevation={3}
                  sx={{
                    width: '100%',
                    bgcolor: '#1E293B',
                    color: '#F8FAFC',
                    borderRadius: 3,
                    p: 2.5,
                    boxShadow: '0 4px 16px rgba(212,175,55,0.10)',
                    border: '2px solid #232946',
                  }}
                >
                  <ListItemText
                    primary={<Typography sx={{ color: '#d4af37', fontWeight: 600, fontSize: '1.08rem' }}>{chat.message}</Typography>}
                    secondary={<Typography sx={{ color: '#F8FAFC', fontWeight: 400 }}>{chat.response}</Typography>}
                  />
                </Paper>
              </ListItem>
            ))
          )}
        </List>
      </Container>
    </Box>
  );
}