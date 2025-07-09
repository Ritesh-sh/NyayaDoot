import React, { useEffect, useState } from 'react';
import { Container, List, ListItem, ListItemText, Typography, Box, Paper } from '@mui/material';

export default function History() {
  const [chats, setChats] = useState([]);

  useEffect(() => {
    const storedHistory = sessionStorage.getItem('chatHistory');
    if (storedHistory) {
      setChats(JSON.parse(storedHistory));
    }
  }, []);

  return (
    <Container>
      <Box my={4}>
        <Typography variant="h4" gutterBottom>
          Chat History
        </Typography>
        <Paper>
          <List>
            {chats.length > 0 ? chats.map((session, idx) => (
              <ListItem key={idx}>
                <ListItemText
                  primary={session.messages[0]?.content?.substring(0, 25) + '...'}
                  secondary={session.last_message_time ? new Date(session.last_message_time).toLocaleString() : 'No timestamp'}
                />
              </ListItem>
            )) : (
              <ListItem>
                <ListItemText primary="No chat history found." />
              </ListItem>
            )}
          </List>
        </Paper>
      </Box>
    </Container>
  );
}