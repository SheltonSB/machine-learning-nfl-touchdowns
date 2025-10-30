import React, { useEffect, useState } from 'react';
import {
    Box,
    Card,
    CardContent,
    Grid,
    Typography,
} from '@mui/material';
import {
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
    LineChart,
    Line,
    CartesianGrid,
    Tooltip,
    Legend,
    XAxis,
    YAxis,
    BarChart,
    Bar,
} from 'recharts';

import { PageHeader, LoadingState, ErrorState } from '../../components/common';
import { api } from '../../services/api';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#9C27B0'];

const overviewCards = [
    { key: 'total_players', label: 'Total Players' },
    { key: 'total_predictions', label: 'Total Predictions' },
    { key: 'accuracy', label: 'Accuracy', isPercentage: true },
    { key: 'active_models', label: 'Active Models' },
];

const formatValue = (value, isPercentage = false) => {
    if (value == null) {
        return '–';
    }
    return isPercentage ? `${(value * 100).toFixed(1)}%` : Intl.NumberFormat().format(value);
};

const transformDistribution = (distribution = {}) =>
    Object.entries(distribution).map(([name, value]) => ({ name, value }));

function AnalyticsPage() {
    const [analytics, setAnalytics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchAnalytics = async () => {
            try {
                setLoading(true);
                const [overview, players, teams, trends] = await Promise.all([
                    api.get('/analytics/overview'),
                    api.get('/analytics/players'),
                    api.get('/analytics/teams'),
                    api.get('/analytics/trends'),
                ]);

                setAnalytics({
                    overview: overview.data,
                    players: players.data,
                    teams: teams.data,
                    trends: trends.data,
                });
            } catch (err) {
                console.error('Failed to load analytics:', err);
                setError('Unable to load analytics data. Please try again later.');
            } finally {
                setLoading(false);
            }
        };

        fetchAnalytics();
    }, []);

    if (loading) {
        return <LoadingState message="Loading analytics..." />;
    }

    if (error) {
        return <ErrorState message={error} />;
    }

    const positionData = transformDistribution(analytics?.players?.position_distribution);
    const teamData = transformDistribution(analytics?.teams?.touchdown_distribution);
    const trendData = analytics?.trends?.weekly_touchdowns ?? [];

    return (
        <Box>
            <PageHeader
                title="Analytics Dashboard"
                subtitle="Track trends across players, teams, and model performance."
            />

            <Grid container spacing={3}>
                {overviewCards.map(({ key, label, isPercentage }) => (
                    <Grid item xs={12} sm={6} md={3} key={key}>
                        <Card>
                            <CardContent>
                                <Typography color="text.secondary" gutterBottom>
                                    {label}
                                </Typography>
                                <Typography variant="h4">
                                    {formatValue(analytics?.overview?.[key], isPercentage)}
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}

                <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Position Distribution
                            </Typography>
                            {positionData.length ? (
                                <ResponsiveContainer width="100%" height={300}>
                                    <PieChart>
                                        <Pie data={positionData} dataKey="value" nameKey="name" cx="50%" cy="50%" label>
                                            {positionData.map((entry, index) => (
                                                <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip />
                                    </PieChart>
                                </ResponsiveContainer>
                            ) : (
                                <Typography color="text.secondary">
                                    No position analytics available.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Team Touchdown Distribution
                            </Typography>
                            {teamData.length ? (
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={teamData}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="name" />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Bar dataKey="value" fill="#1976d2" name="Touchdowns" />
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <Typography color="text.secondary">No team analytics available.</Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Weekly Touchdown Trends
                            </Typography>
                            {trendData.length ? (
                                <ResponsiveContainer width="100%" height={320}>
                                    <LineChart data={trendData}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="week" />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line type="monotone" dataKey="touchdowns" stroke="#ff9800" name="Touchdowns" />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <Typography color="text.secondary">
                                    Not enough data to chart touchdown trends.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
}

export default AnalyticsPage;
