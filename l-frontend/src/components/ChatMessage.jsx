import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Paper, Box } from '@mui/material';

const colors = {
  user: { background: '#d4af37', text: '#0a2463', border: '#c4a030' },
  bot:  { background: '#1E293B', text: '#F8FAFC', border: '#232946' },
  error:{ background: '#2d1b1b', text: '#f87171', border: '#7f1d1d' },
  code: { background: '#232946', text: '#F8FAFC', accent: '#d4af37' },
};

export default function ChatMessage({ answer, type, isError }) {
  const isUser = type === 'user';
  const palette = isError ? colors.error : isUser ? colors.user : colors.bot;

  const bubbleRadius = isUser
    ? { borderTopLeftRadius: 18, borderTopRightRadius: 4, borderBottomLeftRadius: 18, borderBottomRightRadius: 18 }
    : { borderTopLeftRadius: 4, borderTopRightRadius: 18, borderBottomLeftRadius: 18, borderBottomRightRadius: 18 };

  return (
    <Paper
      elevation={isUser ? 2 : 3}
      sx={{
        p: 2, maxWidth: '78%',
        bgcolor: palette.background, color: palette.text,
        border: `1.5px solid ${palette.border}`,
        ...bubbleRadius,
        wordBreak: 'break-word', fontSize: '0.98rem', lineHeight: 1.7,
        fontWeight: 400, boxShadow: isUser ? 1 : 2,
        transition: 'all 0.2s ease', mb: 0.5,
        alignSelf: isUser ? 'flex-end' : 'flex-start',
        '& ul, & ol': { pl: 2.5, my: 0.5 },
        '& li': { mb: 0.3 },
        '& p': { my: 0.3 },
        '& a': { color: '#d4af37', textDecoration: 'underline', '&:hover': { color: '#fff' } },
        '& strong': { color: isUser ? '#0a2463' : '#d4af37' },
      }}
    >
      <ReactMarkdown
        components={{
          p: ({ children }) => <Box component="span" sx={{ display: 'block', mb: 0.5 }}>{children}</Box>,
          ul: ({ children }) => <Box component="ul" sx={{ pl: 2.5, my: 0.5 }}>{children}</Box>,
          ol: ({ children }) => <Box component="ol" sx={{ pl: 2.5, my: 0.5 }}>{children}</Box>,
          li: ({ children }) => <Box component="li" sx={{ mb: 0.2 }}>{children}</Box>,
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer"
              style={{ color: '#d4af37', textDecoration: 'underline' }}
            >{children}</a>
          ),
          code: ({ inline, children, ...props }) =>
            inline ? (
              <code style={{
                backgroundColor: colors.code.accent, color: '#0a2463',
                padding: '1px 6px', borderRadius: 4, fontSize: '0.9em', fontWeight: 600,
              }} {...props}>{children}</code>
            ) : (
              <pre style={{
                backgroundColor: colors.code.background, color: colors.code.text,
                padding: 12, borderRadius: 8, overflowX: 'auto', fontSize: '0.9rem', margin: '4px 0',
              }}><code {...props}>{children}</code></pre>
            ),
        }}
      >
        {answer}
      </ReactMarkdown>
    </Paper>
  );
}
