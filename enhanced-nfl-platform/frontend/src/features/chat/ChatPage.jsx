import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    Alert,
    Box,
    Card,
    CardContent,
    Chip,
    Grid,
    IconButton,
    List,
    ListItem,
    Paper,
    TextField,
    Typography,
} from '@mui/material';
import {
    Send as SendIcon,
    SmartToy as SmartToyIcon,
    Person as PersonIcon,
    Lightbulb as LightbulbIcon,
} from '@mui/icons-material';

import { PageHeader } from '../../components/common';
import { api } from '../../services/api';

const initialMessage = {
    type: 'ai',
    content:
        "Hello! I'm your NFL AI assistant. Ask me anything about players, teams, advanced statistics, or upcoming matchups.",
    timestamp: new Date().toISOString(),
};

function ChatPage() {
    const [messages, setMessages] = useState([initialMessage]);
    const [input, setInput] = useState('');
    const [isSending, setIsSending] = useState(false);
    const [error, setError] = useState(null);
    const [suggestions, setSuggestions] = useState([]);
    const endOfMessagesRef = useRef(null);

    useEffect(() => {
        const fetchSuggestions = async () => {
            try {
                const response = await api.get('/rag/suggestions');
                setSuggestions(response.data.suggestions || []);
            } catch (err) {
                console.error('Failed to fetch suggestions:', err);
            }
        };

        fetchSuggestions();
    }, []);

    useEffect(() => {
        if (endOfMessagesRef.current) {
            endOfMessagesRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    const appendMessage = useCallback((message) => {
        setMessages((prev) => [...prev, { ...message, timestamp: new Date().toISOString() }]);
    }, []);

    const sendMessage = async () => {
        if (!input.trim() || isSending) {
            return;
        }

        const content = input.trim();
        appendMessage({ type: 'user', content });
        setInput('');
        setError(null);
        setIsSending(true);

        try {
            const response = await api.post('/rag/query', {
                question: content,
                top_k: 5,
            });

            appendMessage({
                type: 'ai',
                content: response.data.answer,
                confidence: response.data.confidence,
                relevant_docs: response.data.relevant_docs,
            });
        } catch (err) {
            console.error('Failed to query assistant:', err);
            setError('Unable to reach the AI assistant. Please try again in a moment.');
        } finally {
            setIsSending(false);
        }
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    };

    const renderMessage = (message, index) => {
        const isUser = message.type === 'user';
        return (
            <ListItem
                key={`${message.timestamp}-${index}`}
                sx={{
                    display: 'flex',
                    justifyContent: isUser ? 'flex-end' : 'flex-start',
                }}
            >
                <Box
                    sx={{
                        display: 'flex',
                        gap: 1,
                        alignItems: 'flex-start',
                        maxWidth: '75%',
                        bgcolor: isUser ? 'primary.light' : 'grey.100',
                        color: isUser ? 'common.white' : 'text.primary',
                        borderRadius: 2,
                        p: 2,
                        boxShadow: 1,
                    }}
                >
                    {isUser ? (
                        <PersonIcon fontSize="small" />
                    ) : (
                        <SmartToyIcon fontSize="small" color="primary" />
                    )}
                    <Box>
                        <Typography variant="body1">{message.content}</Typography>
                        <Typography variant="caption" sx={{ display: 'block', opacity: 0.8, mt: 1 }}>
                            {new Date(message.timestamp).toLocaleTimeString([], {
                                hour: '2-digit',
                                minute: '2-digit',
                            })}
                            {message.confidence != null && (
                                <Chip
                                    size="small"
                                    label={`${Math.round(message.confidence * 100)}% confidence`}
                                    color="primary"
                                    variant="outlined"
                                    sx={{ ml: 1 }}
                                />
                            )}
                        </Typography>
                        {message.relevant_docs?.length ? (
                            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                {message.relevant_docs.map((doc, idx) => (
                                    <Typography key={idx} variant="caption" color="text.secondary">
                                        • {doc}
                                    </Typography>
                                ))}
                            </Box>
                        ) : null}
                    </Box>
                </Box>
            </ListItem>
        );
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <PageHeader
                title="AI Chat Assistant"
                subtitle="Use retrieval-augmented responses to explore historical stats and model insights."
            />

            <Grid container spacing={3} sx={{ flexGrow: 1 }}>
                <Grid item xs={12} md={8}>
                    <Paper
                        elevation={2}
                        sx={{
                            height: { xs: '70vh', md: '75vh' },
                            display: 'flex',
                            flexDirection: 'column',
                        }}
                    >
                        <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 2, bgcolor: 'grey.50' }}>
                            <List sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                {messages.map((message, idx) => renderMessage(message, idx))}
                            </List>
                            <div ref={endOfMessagesRef} />
                        </Box>

                        {error && <Alert severity="error">{error}</Alert>}

                        <Box sx={{ display: 'flex', p: 2, gap: 1, alignItems: 'center' }}>
                            <TextField
                                fullWidth
                                multiline
                                maxRows={4}
                                placeholder="Ask about players, matchups, or advanced metrics..."
                                value={input}
                                onChange={(event) => setInput(event.target.value)}
                                onKeyDown={handleKeyPress}
                            />
                            <IconButton
                                color="primary"
                                onClick={sendMessage}
                                disabled={!input.trim() || isSending}
                                size="large"
                                aria-label="send message"
                            >
                                <SendIcon />
                            </IconButton>
                        </Box>
                    </Paper>
                </Grid>

                <Grid item xs={12} md={4}>
                    <Card>
                        <CardContent>
                            <Typography
                                variant="h6"
                                gutterBottom
                                sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
                            >
                                <LightbulbIcon fontSize="small" />
                                Suggested Prompts
                            </Typography>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                {suggestions.length ? (
                                    suggestions.map((suggestion) => (
                                        <Chip
                                            key={suggestion}
                                            label={suggestion}
                                            onClick={() => setInput(suggestion)}
                                            sx={{ justifyContent: 'flex-start' }}
                                        />
                                    ))
                                ) : (
                                    <Typography variant="body2" color="text.secondary">
                                        Suggestions will appear here once available.
                                    </Typography>
                                )}
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
}

export default ChatPage;
