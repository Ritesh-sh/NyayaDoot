import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { Container, List, ListItem, ListItemText, Typography } from '@mui/material';

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
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>
        Chat History
      </Typography>
      <List>
        {chats.map((chat, index) => (
          <ListItem key={index} divider>
            <ListItemText
              primary={chat.message}
              secondary={chat.response}
              secondaryTypographyProps={{ color: "text.primary" }}
              sx={{ bgcolor: '#f5f5f5', borderRadius: 2, p: 2 }}
            />
          </ListItem>
        ))}
      </List>
    </Container>
  );
}