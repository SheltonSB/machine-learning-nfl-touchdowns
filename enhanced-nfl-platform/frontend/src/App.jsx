import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import { AppLayout } from './layouts';
import DashboardPage from './features/dashboard/DashboardPage.jsx';
import PlayersPage from './features/players/PlayersPage.jsx';
import PredictionsPage from './features/predictions/PredictionsPage.jsx';
import AnalyticsPage from './features/analytics/AnalyticsPage.jsx';
import ChatPage from './features/chat/ChatPage.jsx';

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
        h1: { fontSize: '2.5rem', fontWeight: 600 },
        h2: { fontSize: '2rem', fontWeight: 500 },
    },
});

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Router>
                <AppLayout>
                    <Routes>
                        <Route path="/" element={<DashboardPage />} />
                        <Route path="/players" element={<PlayersPage />} />
                        <Route path="/predictions" element={<PredictionsPage />} />
                        <Route path="/analytics" element={<AnalyticsPage />} />
                        <Route path="/chat" element={<ChatPage />} />
                    </Routes>
                </AppLayout>
            </Router>
        </ThemeProvider>
    );
}

export default App;
