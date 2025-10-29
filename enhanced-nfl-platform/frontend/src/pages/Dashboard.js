import React, { useState, useEffect } from 'react';
import {
    Grid,
    Card,
    CardContent,
    Typography,
    Box,
    Chip,
    LinearProgress,
    Alert,
} from '@mui/material';
import {
    SportsFootball,
    TrendingUp,
    People,
    Analytics,
} from '@mui/icons-material';
import { Bar, Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    LineElement,
    PointElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { api } from '../services/api';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    LineElement,
    PointElement,
    Title,
    Tooltip,
    Legend
);

const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [recentPredictions, setRecentPredictions] = useState([]);
    const [modelPerformance, setModelPerformance] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async() => {
        try {
            setLoading(true);
            const [statsData, predictionsData, performanceData] = await Promise.all([
                api.get('/analytics/overview'),
                api.get('/predictions?limit=5'),
                api.get('/predictions/model/performance'),
            ]);

            setStats(statsData.data);
            setRecentPredictions(predictionsData.data);
            setModelPerformance(performanceData.data);
        } catch (err) {
            setError('Failed to load dashboard data');
            console.error('Dashboard error:', err);
        } finally {
            setLoading(false);
        }
    };

    const getPerformanceChartData = () => {
        if (!modelPerformance) return null;

        return {
            labels: Object.keys(modelPerformance),
            datasets: [{
                    label: 'Accuracy',
                    data: Object.values(modelPerformance).map(model => model.accuracy),
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1,
                },
                {
                    label: 'F1 Score',
                    data: Object.values(modelPerformance).map(model => model.f1_score),
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1,
                },
            ],
        };
    };

    const getPredictionsChartData = () => {
        if (!recentPredictions.length) return null;

        const last7Days = recentPredictions.slice(0, 7);
        return {
            labels: last7Days.map((_, index) => `Day ${index + 1}`),
            datasets: [{
                label: 'Predictions',
                data: last7Days.map(p => p.prediction ? 1 : 0),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
            }, ],
        };
    };

    if (loading) {
        return ( <
            Box sx = {
                { width: '100%', mt: 2 } } >
            <
            LinearProgress / >
            <
            Typography variant = "h6"
            sx = {
                { mt: 2, textAlign: 'center' } } >
            Loading dashboard... <
            /Typography> <
            /Box>
        );
    }

    if (error) {
        return ( <
            Alert severity = "error"
            sx = {
                { mt: 2 } } > { error } <
            /Alert>
        );
    }

    return ( <
        Box >
        <
        Typography variant = "h4"
        component = "h1"
        gutterBottom >
        NFL AI Platform Dashboard <
        /Typography>

        <
        Grid container spacing = { 3 } > { /* Stats Cards */ } <
        Grid item xs = { 12 }
        sm = { 6 }
        md = { 3 } >
        <
        Card >
        <
        CardContent >
        <
        Box display = "flex"
        alignItems = "center" >
        <
        People color = "primary"
        sx = {
            { fontSize: 40, mr: 2 } }
        /> <
        Box >
        <
        Typography color = "textSecondary"
        gutterBottom >
        Total Players <
        /Typography> <
        Typography variant = "h4" > { stats ? .total_players || 0 } <
        /Typography> <
        /Box> <
        /Box> <
        /CardContent> <
        /Card> <
        /Grid>

        <
        Grid item xs = { 12 }
        sm = { 6 }
        md = { 3 } >
        <
        Card >
        <
        CardContent >
        <
        Box display = "flex"
        alignItems = "center" >
        <
        SportsFootball color = "secondary"
        sx = {
            { fontSize: 40, mr: 2 } }
        /> <
        Box >
        <
        Typography color = "textSecondary"
        gutterBottom >
        Predictions Made <
        /Typography> <
        Typography variant = "h4" > { stats ? .total_predictions || 0 } <
        /Typography> <
        /Box> <
        /Box> <
        /CardContent> <
        /Card> <
        /Grid>

        <
        Grid item xs = { 12 }
        sm = { 6 }
        md = { 3 } >
        <
        Card >
        <
        CardContent >
        <
        Box display = "flex"
        alignItems = "center" >
        <
        TrendingUp color = "success"
        sx = {
            { fontSize: 40, mr: 2 } }
        /> <
        Box >
        <
        Typography color = "textSecondary"
        gutterBottom >
        Accuracy <
        /Typography> <
        Typography variant = "h4" > { stats ? .accuracy ? `${(stats.accuracy * 100).toFixed(1)}%` : 'N/A' } <
        /Typography> <
        /Box> <
        /Box> <
        /CardContent> <
        /Card> <
        /Grid>

        <
        Grid item xs = { 12 }
        sm = { 6 }
        md = { 3 } >
        <
        Card >
        <
        CardContent >
        <
        Box display = "flex"
        alignItems = "center" >
        <
        Analytics color = "warning"
        sx = {
            { fontSize: 40, mr: 2 } }
        /> <
        Box >
        <
        Typography color = "textSecondary"
        gutterBottom >
        Active Models <
        /Typography> <
        Typography variant = "h4" > { modelPerformance ? Object.keys(modelPerformance).length : 0 } <
        /Typography> <
        /Box> <
        /Box> <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Model Performance Chart */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Model Performance <
        /Typography> {
            getPerformanceChartData() ? ( <
                Bar data = { getPerformanceChartData() }
                options = {
                    {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'ML Model Comparison',
                            },
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                            },
                        },
                    }
                }
                />
            ) : ( <
                Typography > No performance data available < /Typography>
            )
        } <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Recent Predictions Chart */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Recent Predictions <
        /Typography> {
            getPredictionsChartData() ? ( <
                Line data = { getPredictionsChartData() }
                options = {
                    {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Last 7 Days',
                            },
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                            },
                        },
                    }
                }
                />
            ) : ( <
                Typography > No prediction data available < /Typography>
            )
        } <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Recent Predictions List */ } <
        Grid item xs = { 12 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Recent Predictions <
        /Typography> {
            recentPredictions.length > 0 ? (
                recentPredictions.map((prediction, index) => ( <
                    Box key = { index }
                    display = "flex"
                    justifyContent = "space-between"
                    alignItems = "center"
                    py = { 1 }
                    borderBottom = { index < recentPredictions.length - 1 ? 1 : 0 }
                    borderColor = "divider" >
                    <
                    Typography > { prediction.player ? .first_name } { prediction.player ? .last_name } <
                    /Typography> <
                    Chip label = { prediction.prediction ? 'TD Predicted' : 'No TD' }
                    color = { prediction.prediction ? 'success' : 'error' }
                    size = "small" /
                    >
                    <
                    Typography variant = "body2"
                    color = "textSecondary" > {
                        (prediction.confidence * 100).toFixed(1) } % confidence <
                    /Typography> <
                    /Box>
                ))
            ) : ( <
                Typography > No recent predictions < /Typography>
            )
        } <
        /CardContent> <
        /Card> <
        /Grid> <
        /Grid> <
        /Box>
    );
};

export default Dashboard;
