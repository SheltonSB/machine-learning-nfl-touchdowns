import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Paper,
    TextField,
    IconButton,
    Typography,
    List,
    ListItem,
    ListItemText,
    Chip,
    CircularProgress,
    Alert,
    Card,
    CardContent,
    Grid,
} from '@mui/material';
import {
    Send,
    SmartToy,
    Person,
    Lightbulb,
} from '@mui/icons-material';
import { api } from '../services/api';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [suggestions, setSuggestions] = useState([]);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        loadSuggestions();
        // Add welcome message
        setMessages([{
            type: 'ai',
            content: 'Hello! I\'m your NFL AI assistant. Ask me anything about players, teams, statistics, or predictions!',
            timestamp: new Date(),
        }, ]);
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current ? .scrollIntoView({ behavior: 'smooth' });
    };

    const loadSuggestions = async() => {
        try {
            const response = await api.get('/rag/suggestions');
            setSuggestions(response.data.suggestions);
        } catch (err) {
            console.error('Error loading suggestions:', err);
        }
    };

    const handleSendMessage = async() => {
        if (!input.trim() || loading) return;

        const userMessage = {
            type: 'user',
            content: input.trim(),
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);
        setError(null);

        try {
            const response = await api.post('/rag/query', {
                question: userMessage.content,
                top_k: 5,
            });

            const aiMessage = {
                type: 'ai',
                content: response.data.answer,
                timestamp: new Date(),
                confidence: response.data.confidence,
                relevant_docs: response.data.relevant_docs,
            };

            setMessages(prev => [...prev, aiMessage]);
        } catch (err) {
            setError('Failed to get response from AI assistant');
            console.error('Chat error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    };

    const handleSuggestionClick = (suggestion) => {
        setInput(suggestion);
    };

    const formatTimestamp = (timestamp) => {
        return new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return ( <
        Box sx = {
            { height: '100%', display: 'flex', flexDirection: 'column' } } >
        <
        Typography variant = "h4"
        component = "h1"
        gutterBottom >
        AI Chat Assistant <
        /Typography>

        <
        Grid container spacing = { 3 }
        sx = {
            { flexGrow: 1 } } > { /* Chat Area */ } <
        Grid item xs = { 12 }
        md = { 8 } >
        <
        Paper elevation = { 2 }
        sx = {
            {
                height: '70vh',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
            }
        } >
        { /* Messages */ } <
        Box sx = {
            {
                flexGrow: 1,
                overflow: 'auto',
                p: 2,
                backgroundColor: '#fafafa',
            }
        } >
        <
        List > {
            messages.map((message, index) => ( <
                ListItem key = { index }
                sx = {
                    {
                        flexDirection: message.type === 'user' ? 'row-reverse' : 'row',
                        alignItems: 'flex-start',
                    }
                } >
                <
                Box sx = {
                    {
                        display: 'flex',
                        alignItems: 'center',
                        mb: 1,
                        maxWidth: '70%',
                        backgroundColor: message.type === 'user' ? '#e3f2fd' : '#f5f5f5',
                        borderRadius: 2,
                        p: 2,
                        ml: message.type === 'user' ? 2 : 0,
                        mr: message.type === 'user' ? 0 : 2,
                    }
                } >
                {
                    message.type === 'ai' ? ( <
                        SmartToy color = "primary"
                        sx = {
                            { mr: 1 } }
                        />
                    ) : ( <
                        Person color = "primary"
                        sx = {
                            { mr: 1 } }
                        />
                    )
                } <
                Box sx = {
                    { flexGrow: 1 } } >
                <
                Typography variant = "body1" > { message.content } <
                /Typography> <
                Typography variant = "caption"
                color = "textSecondary"
                sx = {
                    { display: 'block', mt: 1 } } >
                { formatTimestamp(message.timestamp) } {
                    message.confidence && ( <
                        Chip label = { `${(message.confidence * 100).toFixed(0)}% confidence` }
                        size = "small"
                        color = "primary"
                        sx = {
                            { ml: 1 } }
                        />
                    )
                } <
                /Typography> <
                /Box> <
                /Box> <
                /ListItem>
            ))
        } {
            loading && ( <
                ListItem >
                <
                Box sx = {
                    { display: 'flex', alignItems: 'center' } } >
                <
                SmartToy color = "primary"
                sx = {
                    { mr: 1 } }
                /> <
                CircularProgress size = { 20 }
                sx = {
                    { mr: 1 } }
                /> <
                Typography > AI is thinking... < /Typography> <
                /Box> <
                /ListItem>
            )
        } <
        div ref = { messagesEndRef }
        /> <
        /List> <
        /Box>

        { /* Input Area */ } <
        Box sx = {
            { p: 2, borderTop: 1, borderColor: 'divider' } } > {
            error && ( <
                Alert severity = "error"
                sx = {
                    { mb: 2 } } > { error } <
                /Alert>
            )
        } <
        Box sx = {
            { display: 'flex', gap: 1 } } >
        <
        TextField fullWidth multiline maxRows = { 4 }
        value = { input }
        onChange = {
            (e) => setInput(e.target.value) }
        onKeyPress = { handleKeyPress }
        placeholder = "Ask me anything about NFL data, players, or predictions..."
        disabled = { loading }
        /> <
        IconButton color = "primary"
        onClick = { handleSendMessage }
        disabled = {!input.trim() || loading } >
        <
        Send / >
        <
        /IconButton> <
        /Box> <
        /Box> <
        /Paper> <
        /Grid>

        { /* Suggestions Sidebar */ } <
        Grid item xs = { 12 }
        md = { 4 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        <
        Lightbulb sx = {
            { mr: 1, verticalAlign: 'middle' } }
        />
        Suggested Questions <
        /Typography> <
        Box sx = {
            { display: 'flex', flexDirection: 'column', gap: 1 } } > {
            suggestions.map((suggestion, index) => ( <
                Chip key = { index }
                label = { suggestion }
                onClick = {
                    () => handleSuggestionClick(suggestion) }
                variant = "outlined"
                sx = {
                    {
                        textAlign: 'left',
                        justifyContent: 'flex-start',
                        height: 'auto',
                        py: 1,
                        '& .MuiChip-label': {
                            whiteSpace: 'normal',
                        },
                    }
                }
                />
            ))
        } <
        /Box> <
        /CardContent> <
        /Card>

        { /* Chat Stats */ } <
        Card sx = {
            { mt: 2 } } >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Chat Statistics <
        /Typography> <
        Typography variant = "body2"
        color = "textSecondary" >
        Messages: { messages.length } <
        /Typography> <
        Typography variant = "body2"
        color = "textSecondary" >
        AI Responses: { messages.filter(m => m.type === 'ai').length } <
        /Typography> <
        /CardContent> <
        /Card> <
        /Grid> <
        /Grid> <
        /Box>
    );
};

export default Chat;
