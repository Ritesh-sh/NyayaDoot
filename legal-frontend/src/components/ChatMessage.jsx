import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Paper } from '@mui/material';

export default function ChatMessage({ answer, type }) {
  const isUser = type === 'user';

  // Custom color palette
  const colors = {
    user: {
      background: '#7C3AED', // violet
      text: '#FFFFFF',
    },
    bot: {
      background: '#1E293B', // deep blue
      text: '#E5E7EB', // light gray
    },
    codeBlock: {
      background: '#0F0F0F', // near-black
      text: '#D1D5DB', // subtle gray for code text
      accent: '#7C3AED', // violet accent for inline code
    }
  };

  const bubbleStyles = isUser
    ? { borderTopLeftRadius: 16, borderTopRightRadius: 0, borderBottomLeftRadius: 16, borderBottomRightRadius: 16 }
    : { borderTopLeftRadius: 0, borderTopRightRadius: 16, borderBottomLeftRadius: 16, borderBottomRightRadius: 16 };

  return (
    <Paper
      elevation={3}
      sx={{
        p: 2,
        maxWidth: '75%',
        bgcolor: isUser ? colors.user.background : colors.bot.background,
        color: isUser ? colors.user.text : colors.bot.text,
        ...bubbleStyles,
        wordBreak: 'break-word',
        whiteSpace: 'pre-wrap',
        fontSize: '0.95rem',
        lineHeight: 1.6,
        boxShadow: 3,
        transition: 'all 0.3s ease-in-out'
      }}
    >
      <ReactMarkdown
        components={{
          p: ({ node, ...props }) => <span {...props} />,
          code: ({ node, inline, className, ...props }) =>
            inline ? (
              <code
                style={{
                  backgroundColor: colors.codeBlock.accent,
                  color: colors.user.text,
                  padding: '2px 6px',
                  borderRadius: 4,
                  fontSize: '0.85em'
                }}
                {...props}
              />
            ) : (
              <pre
                style={{
                  backgroundColor: colors.codeBlock.background,
                  color: colors.codeBlock.text,
                  padding: '12px',
                  borderRadius: 8,
                  overflowX: 'auto',
                  fontSize: '0.85rem'
                }}
              >
                <code {...props} />
              </pre>
            )
        }}
      >
        {answer}
      </ReactMarkdown>
    </Paper>
  );
}
