import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Paper } from '@mui/material';

export default function ChatMessage({ answer, type }) {
  const isUser = type === 'user';

  // Updated color palette for modern gold/blue style
  const colors = {
    user: {
      background: '#d4af37', // gold
      text: '#0a2463', // deep blue text
      border: '#c4a030',
    },
    bot: {
      background: '#1E293B', // deep blue
      text: '#F8FAFC', // light text
      border: '#232946',
    },
    codeBlock: {
      background: '#232946', // dark blue for code
      text: '#F8FAFC', // light text for code
      accent: '#d4af37', // gold accent for inline code
    }
  };

  const bubbleStyles = isUser
    ? {
        borderTopLeftRadius: 18,
        borderTopRightRadius: 0,
        borderBottomLeftRadius: 18,
        borderBottomRightRadius: 18,
        border: `2px solid ${colors.user.border}`,
        alignSelf: 'flex-end',
      }
    : {
        borderTopLeftRadius: 0,
        borderTopRightRadius: 18,
        borderBottomLeftRadius: 18,
        borderBottomRightRadius: 18,
        border: `2px solid ${colors.bot.border}`,
        alignSelf: 'flex-start',
      };

  return (
    <Paper
      elevation={4}
      sx={{
        p: 2.2,
        maxWidth: '75%',
        bgcolor: isUser ? colors.user.background : colors.bot.background,
        color: isUser ? colors.user.text : colors.bot.text,
        ...bubbleStyles,
        wordBreak: 'break-word',
        whiteSpace: 'pre-wrap',
        fontSize: '1.05rem',
        lineHeight: 1.7,
        fontWeight: 500,
        boxShadow: 4,
        transition: 'all 0.3s cubic-bezier(.4,2,.6,1)',
        mb: 0.5,
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
                  padding: '2px 8px',
                  borderRadius: 4,
                  fontSize: '0.93em',
                  fontWeight: 600,
                }}
                {...props}
              />
            ) : (
              <pre
                style={{
                  backgroundColor: colors.codeBlock.background,
                  color: colors.codeBlock.text,
                  padding: '14px',
                  borderRadius: 8,
                  overflowX: 'auto',
                  fontSize: '0.97rem',
                  margin: 0,
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
