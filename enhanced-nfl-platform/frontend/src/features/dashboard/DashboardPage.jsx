import React, { useEffect, useState } from 'react';
import {
    Grid,
    Card,
    CardContent,
    Typography,
    Box,
    Chip,
} from '@mui/material';
import {
    People as PeopleIcon,
    SportsFootball as SportsFootballIcon,
    TrendingUp as TrendingUpIcon,
    Analytics as AnalyticsIcon,
} from '@mui/icons-material';
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
import { Bar, Line } from 'react-chartjs-2';

import { PageHeader, LoadingState, ErrorState } from '../../components/common';
import { api } from '../../services/api';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    LineElement,
    PointElement,
    Title,
    Tooltip,
    Legend,
);

const statCards = [
    {
        key: 'total_players',
        label: 'Total Players',
        icon: <PeopleIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />,
    },
    {
        key: 'total_predictions',
        label: 'Predictions Made',
        icon: <SportsFootballIcon color="secondary" sx={{ fontSize: 40, mr: 2 }} />,
    },
    {
        key: 'accuracy',
        label: 'Overall Accuracy',
        icon: <TrendingUpIcon sx={{ fontSize: 40, mr: 2, color: 'success.main' }} />,
        isPercentage: true,
    },
    {
        key: 'active_models',
        label: 'Active Models',
        icon: <AnalyticsIcon sx={{ fontSize: 40, mr: 2, color: 'info.main' }} />,
    },
];

const formatStatValue = (value, isPercentage = false) => {
    if (value == null) {
        return '–';
    }
    if (isPercentage) {
        return `${(value * 100).toFixed(1)}%`;
    }
    return Intl.NumberFormat().format(value);
};

const buildPerformanceChart = (performance) => {
    if (!performance) {
        return null;
    }
    const labels = Object.keys(performance);
    return {
        labels,
        datasets: [
            {
                label: 'Accuracy',
                data: labels.map((model) => performance[model].accuracy ?? 0),
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
            },
            {
                label: 'F1 Score',
                data: labels.map((model) => performance[model].f1_score ?? 0),
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
            },
        ],
    };
};

const buildPredictionTrend = (predictions) => {
    if (!predictions?.length) {
        return null;
    }
    const recent = predictions.slice(0, 7);
    return {
        labels: recent.map((_, index) => `Day ${index + 1}`),
        datasets: [
            {
                label: 'Predictions',
                data: recent.map((prediction) => (prediction.prediction ? 1 : 0)),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
            },
        ],
    };
};

function DashboardPage() {
    const [stats, setStats] = useState(null);
    const [recentPredictions, setRecentPredictions] = useState([]);
    const [modelPerformance, setModelPerformance] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchDashboardData = async () => {
            try {
                setLoading(true);
                const [statsResponse, predictionsResponse, performanceResponse] = await Promise.all([
                    api.get('/analytics/overview'),
                    api.get('/predictions?limit=5'),
                    api.get('/predictions/model/performance'),
                ]);
                setStats(statsResponse.data);
                setRecentPredictions(predictionsResponse.data);
                setModelPerformance(performanceResponse.data);
            } catch (err) {
                console.error('Failed to load dashboard data:', err);
                setError('Unable to load dashboard data. Please try again later.');
            } finally {
                setLoading(false);
            }
        };

        fetchDashboardData();
    }, []);

    if (loading) {
        return <LoadingState message="Loading dashboard..." />;
    }

    if (error) {
        return <ErrorState message={error} />;
    }

    const performanceChart = buildPerformanceChart(modelPerformance);
    const predictionTrend = buildPredictionTrend(recentPredictions);

    return (
        <Box>
            <PageHeader
                title="NFL AI Platform Dashboard"
                subtitle="Overview of recent model performance and system activity"
            />

            <Grid container spacing={3}>
                {statCards.map(({ key, label, icon, isPercentage }) => (
                    <Grid item xs={12} sm={6} md={3} key={key}>
                        <Card>
                            <CardContent>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    {icon}
                                    <Box>
                                        <Typography color="text.secondary" gutterBottom>
                                            {label}
                                        </Typography>
                                        <Typography variant="h4">
                                            {formatStatValue(stats?.[key], isPercentage)}
                                        </Typography>
                                    </Box>
                                </Box>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}

                <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Model Performance
                            </Typography>
                            {performanceChart ? (
                                <Bar data={performanceChart} />
                            ) : (
                                <Typography color="text.secondary">
                                    Performance data unavailable.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Prediction Trend
                            </Typography>
                            {predictionTrend ? (
                                <Line data={predictionTrend} />
                            ) : (
                                <Typography color="text.secondary">
                                    Not enough prediction history to plot a trend.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Recent Predictions
                            </Typography>
                            {recentPredictions.length ? (
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                    {recentPredictions.map((prediction) => (
                                        <Box
                                            key={prediction.id || prediction.game_id}
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'space-between',
                                                p: 2,
                                                borderRadius: 1,
                                                bgcolor: 'background.paper',
                                            }}
                                        >
                                            <Box>
                                                <Typography variant="subtitle1">
                                                    {prediction.player_name}
                                                </Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    {prediction.game_date} vs {prediction.opponent}
                                                </Typography>
                                            </Box>
                                            <Chip
                                                label={
                                                    prediction.prediction ? 'Touchdown' : 'No Touchdown'
                                                }
                                                color={prediction.prediction ? 'success' : 'default'}
                                            />
                                        </Box>
                                    ))}
                                </Box>
                            ) : (
                                <Typography color="text.secondary">
                                    No recent predictions available.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
}

export default DashboardPage;
