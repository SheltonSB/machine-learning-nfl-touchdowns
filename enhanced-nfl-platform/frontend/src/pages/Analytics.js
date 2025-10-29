import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    Grid,
    CircularProgress,
    Alert,
} from '@mui/material';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    LineChart,
    Line,
    PieChart,
    Pie,
    Cell,
} from 'recharts';
import { api } from '../services/api';

const Analytics = () => {
    const [analyticsData, setAnalyticsData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        loadAnalyticsData();
    }, []);

    const loadAnalyticsData = async() => {
        try {
            setLoading(true);
            const [overview, players, teams, trends] = await Promise.all([
                api.get('/analytics/overview'),
                api.get('/analytics/players'),
                api.get('/analytics/teams'),
                api.get('/analytics/trends'),
            ]);

            setAnalyticsData({
                overview: overview.data,
                players: players.data,
                teams: teams.data,
                trends: trends.data,
            });
        } catch (err) {
            setError('Failed to load analytics data');
            console.error('Analytics error:', err);
        } finally {
            setLoading(false);
        }
    };

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

    if (loading) {
        return ( <
            Box display = "flex"
            justifyContent = "center"
            alignItems = "center"
            minHeight = "400px" >
            <
            CircularProgress / >
            <
            Typography variant = "h6"
            sx = {
                { ml: 2 } } >
            Loading analytics... <
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
        Analytics Dashboard <
        /Typography>

        <
        Grid container spacing = { 3 } > { /* Overview Cards */ } <
        Grid item xs = { 12 }
        sm = { 6 }
        md = { 3 } >
        <
        Card >
        <
        CardContent >
        <
        Typography color = "textSecondary"
        gutterBottom >
        Total Players <
        /Typography> <
        Typography variant = "h4" > { analyticsData ? .overview ? .total_players || 0 } <
        /Typography> <
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
        Typography color = "textSecondary"
        gutterBottom >
        Total Predictions <
        /Typography> <
        Typography variant = "h4" > { analyticsData ? .overview ? .total_predictions || 0 } <
        /Typography> <
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
        Typography color = "textSecondary"
        gutterBottom >
        Accuracy <
        /Typography> <
        Typography variant = "h4" > {
            analyticsData ? .overview ? .accuracy ?
            `${(analyticsData.overview.accuracy * 100).toFixed(1)}%` :
            'N/A'
        } <
        /Typography> <
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
        Typography color = "textSecondary"
        gutterBottom >
        Active Models <
        /Typography> <
        Typography variant = "h4" > { analyticsData ? .overview ? .active_models || 0 } <
        /Typography> <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Position Distribution Chart */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Position Distribution <
        /Typography> <
        ResponsiveContainer width = "100%"
        height = { 300 } >
        <
        PieChart >
        <
        Pie data = { Object.entries(analyticsData ? .players ? .position_distribution || {}).map(([name, value]) => ({ name, value })) }
        cx = "50%"
        cy = "50%"
        labelLine = { false }
        label = {
            ({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%` }
        outerRadius = { 80 }
        fill = "#8884d8"
        dataKey = "value" >
        {
            Object.entries(analyticsData ? .players ? .position_distribution || {}).map((entry, index) => ( <
                Cell key = { `cell-${index}` }
                fill = { COLORS[index % COLORS.length] }
                />
            ))
        } <
        /Pie> <
        Tooltip / >
        <
        /PieChart> <
        /ResponsiveContainer> <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Conference Distribution Chart */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Conference Distribution <
        /Typography> <
        ResponsiveContainer width = "100%"
        height = { 300 } >
        <
        BarChart data = { Object.entries(analyticsData ? .teams ? .conference_breakdown || {}).map(([name, value]) => ({ name, value })) } >
        <
        CartesianGrid strokeDasharray = "3 3" / >
        <
        XAxis dataKey = "name" / >
        <
        YAxis / >
        <
        Tooltip / >
        <
        Bar dataKey = "value"
        fill = "#8884d8" / >
        <
        /BarChart> <
        /ResponsiveContainer> <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Top Performers Chart */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Top Performers(Touchdowns) <
        /Typography> <
        ResponsiveContainer width = "100%"
        height = { 300 } >
        <
        BarChart data = { analyticsData ? .players ? .top_performers || [] } >
        <
        CartesianGrid strokeDasharray = "3 3" / >
        <
        XAxis dataKey = "name" / >
        <
        YAxis / >
        <
        Tooltip / >
        <
        Legend / >
        <
        Bar dataKey = "touchdowns"
        fill = "#8884d8"
        name = "Touchdowns" / >
        <
        Bar dataKey = "yards"
        fill = "#82ca9d"
        name = "Yards" / >
        <
        /BarChart> <
        /ResponsiveContainer> <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Prediction Accuracy Trend */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Prediction Accuracy Trend <
        /Typography> <
        ResponsiveContainer width = "100%"
        height = { 300 } >
        <
        LineChart data = { analyticsData ? .trends ? .prediction_accuracy ? .monthly ? .map((value, index) => ({ month: `Month ${index + 1}`, accuracy: value })) || [] } >
        <
        CartesianGrid strokeDasharray = "3 3" / >
        <
        XAxis dataKey = "month" / >
        <
        YAxis domain = {
            [0, 1] }
        /> <
        Tooltip / >
        <
        Line type = "monotone"
        dataKey = "accuracy"
        stroke = "#8884d8"
        strokeWidth = { 2 }
        /> <
        /LineChart> <
        /ResponsiveContainer> <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Passing Trends */ } <
        Grid item xs = { 12 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Passing Trends Over Years <
        /Typography> <
        ResponsiveContainer width = "100%"
        height = { 400 } >
        <
        LineChart data = { Object.entries(analyticsData ? .trends ? .passing_trends || {}).map(([year, data]) => ({ year, ...data })) } >
        <
        CartesianGrid strokeDasharray = "3 3" / >
        <
        XAxis dataKey = "year" / >
        <
        YAxis yAxisId = "left" / >
        <
        YAxis yAxisId = "right"
        orientation = "right" / >
        <
        Tooltip / >
        <
        Legend / >
        <
        Bar yAxisId = "left"
        dataKey = "avg_yards"
        fill = "#8884d8"
        name = "Avg Yards" / >
        <
        Line yAxisId = "right"
        type = "monotone"
        dataKey = "avg_tds"
        stroke = "#82ca9d"
        strokeWidth = { 2 }
        name = "Avg TDs" / >
        <
        /LineChart> <
        /ResponsiveContainer> <
        /CardContent> <
        /Card> <
        /Grid> <
        /Grid> <
        /Box>
    );
};

export default Analytics;
