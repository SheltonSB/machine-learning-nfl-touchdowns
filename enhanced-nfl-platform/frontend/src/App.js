import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

// Components
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Players from './pages/Players';
import Predictions from './pages/Predictions';
import Analytics from './pages/Analytics';
import Chat from './pages/Chat';

// Create theme
const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#1976d2',
        },
        secondary: {
            main: '#dc004e',
        },
        background: {
            default: '#f5f5f5',
        },
    },
    typography: {
        h1: {
            fontSize: '2.5rem',
            fontWeight: 600,
        },
        h2: {
            fontSize: '2rem',
            fontWeight: 500,
        },
    },
});

function App() {
    return ( <
        ThemeProvider theme = { theme } >
        <
        CssBaseline / >
        <
        Router >
        <
        Box sx = {
            { display: 'flex', flexDirection: 'column', minHeight: '100vh' } } >
        <
        Navbar / >
        <
        Box component = "main"
        sx = {
            { flexGrow: 1, p: 3 } } >
        <
        Routes >
        <
        Route path = "/"
        element = { < Dashboard / > }
        /> <
        Route path = "/players"
        element = { < Players / > }
        /> <
        Route path = "/predictions"
        element = { < Predictions / > }
        /> <
        Route path = "/analytics"
        element = { < Analytics / > }
        /> <
        Route path = "/chat"
        element = { < Chat / > }
        /> <
        /Routes> <
        /Box> <
        /Box> <
        /Router> <
        /ThemeProvider>
    );
}

export default App;
