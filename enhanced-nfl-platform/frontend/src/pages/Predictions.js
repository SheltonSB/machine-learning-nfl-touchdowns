import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    Grid,
    Button,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Chip,
    Alert,
    CircularProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
} from '@mui/material';
import {
    SportsFootball,
    TrendingUp,
    Psychology,
} from '@mui/icons-material';
import { api } from '../services/api';

const Predictions = () => {
    const [players, setPlayers] = useState([]);
    const [selectedPlayer, setSelectedPlayer] = useState('');
    const [features, setFeatures] = useState({
        passing_yards_roll3: 250,
        td_passes_roll3: 1.5,
        passes_attempted_roll3: 35,
        age: 28,
        experience: 5,
        height: 74,
        weight: 220
    });
    const [modelName, setModelName] = useState('ensemble');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [recentPredictions, setRecentPredictions] = useState([]);

    useEffect(() => {
        loadPlayers();
        loadRecentPredictions();
    }, []);

    const loadPlayers = async() => {
        try {
            const response = await api.get('/players?limit=50');
            setPlayers(response.data);
        } catch (error) {
            console.error('Error loading players:', error);
        }
    };

    const loadRecentPredictions = async() => {
        try {
            const response = await api.get('/predictions?limit=10');
            setRecentPredictions(response.data);
        } catch (error) {
            console.error('Error loading predictions:', error);
        }
    };

    const handleMakePrediction = async() => {
        if (!selectedPlayer) {
            alert('Please select a player');
            return;
        }

        try {
            setLoading(true);
            const response = await api.post('/predictions', {
                player_id: parseInt(selectedPlayer),
                features: features,
                model_name: modelName
            });
            setPrediction(response.data);
            loadRecentPredictions(); // Refresh recent predictions
        } catch (error) {
            console.error('Error making prediction:', error);
            alert('Error making prediction. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleFeatureChange = (feature, value) => {
        setFeatures(prev => ({
            ...prev,
            [feature]: parseFloat(value) || 0
        }));
    };

    return ( <
        Box >
        <
        Typography variant = "h4"
        component = "h1"
        gutterBottom >
        Touchdown Predictions <
        /Typography>

        <
        Grid container spacing = { 3 } > { /* Prediction Form */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        <
        SportsFootball sx = {
            { mr: 1, verticalAlign: 'middle' } }
        />
        Make Prediction <
        /Typography>

        <
        Grid container spacing = { 2 } >
        <
        Grid item xs = { 12 } >
        <
        FormControl fullWidth >
        <
        InputLabel > Select Player < /InputLabel> <
        Select value = { selectedPlayer }
        onChange = {
            (e) => setSelectedPlayer(e.target.value) } >
        {
            players.map((player) => ( <
                MenuItem key = { player.id }
                value = { player.id } > { player.first_name } { player.last_name }({ player.position }) <
                /MenuItem>
            ))
        } <
        /Select> <
        /FormControl> <
        /Grid>

        <
        Grid item xs = { 12 } >
        <
        FormControl fullWidth >
        <
        InputLabel > ML Model < /InputLabel> <
        Select value = { modelName }
        onChange = {
            (e) => setModelName(e.target.value) } >
        <
        MenuItem value = "ensemble" > Ensemble(Recommended) < /MenuItem> <
        MenuItem value = "xgboost" > XGBoost < /MenuItem> <
        MenuItem value = "tensorflow" > TensorFlow < /MenuItem> <
        MenuItem value = "pytorch" > PyTorch < /MenuItem> <
        /Select> <
        /FormControl> <
        /Grid>

        <
        Grid item xs = { 12 } >
        <
        Typography variant = "subtitle2"
        gutterBottom >
        Player Features <
        /Typography> <
        /Grid>

        <
        Grid item xs = { 6 } >
        <
        TextField fullWidth label = "Passing Yards (avg)"
        type = "number"
        value = { features.passing_yards_roll3 }
        onChange = {
            (e) => handleFeatureChange('passing_yards_roll3', e.target.value) }
        /> <
        /Grid>

        <
        Grid item xs = { 6 } >
        <
        TextField fullWidth label = "TD Passes (avg)"
        type = "number"
        step = "0.1"
        value = { features.td_passes_roll3 }
        onChange = {
            (e) => handleFeatureChange('td_passes_roll3', e.target.value) }
        /> <
        /Grid>

        <
        Grid item xs = { 6 } >
        <
        TextField fullWidth label = "Pass Attempts (avg)"
        type = "number"
        value = { features.passes_attempted_roll3 }
        onChange = {
            (e) => handleFeatureChange('passes_attempted_roll3', e.target.value) }
        /> <
        /Grid>

        <
        Grid item xs = { 6 } >
        <
        TextField fullWidth label = "Age"
        type = "number"
        value = { features.age }
        onChange = {
            (e) => handleFeatureChange('age', e.target.value) }
        /> <
        /Grid>

        <
        Grid item xs = { 6 } >
        <
        TextField fullWidth label = "Experience (years)"
        type = "number"
        value = { features.experience }
        onChange = {
            (e) => handleFeatureChange('experience', e.target.value) }
        /> <
        /Grid>

        <
        Grid item xs = { 6 } >
        <
        TextField fullWidth label = "Height (inches)"
        type = "number"
        value = { features.height }
        onChange = {
            (e) => handleFeatureChange('height', e.target.value) }
        /> <
        /Grid>

        <
        Grid item xs = { 12 } >
        <
        Button fullWidth variant = "contained"
        size = "large"
        onClick = { handleMakePrediction }
        disabled = { loading }
        startIcon = { loading ? < CircularProgress size = { 20 } /> : <Psychology / > } >
        { loading ? 'Making Prediction...' : 'Predict Touchdown' } <
        /Button> <
        /Grid> <
        /Grid> <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Prediction Result */ } <
        Grid item xs = { 12 }
        md = { 6 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        <
        TrendingUp sx = {
            { mr: 1, verticalAlign: 'middle' } }
        />
        Prediction Result <
        /Typography>

        {
            prediction ? ( <
                Box >
                <
                Alert severity = { prediction.prediction ? 'success' : 'error' }
                sx = {
                    { mb: 2 } } >
                <
                Typography variant = "h6" > { prediction.prediction ? 'TOUCHDOWN PREDICTED!' : 'No Touchdown Predicted' } <
                /Typography> <
                /Alert>

                <
                Grid container spacing = { 2 } >
                <
                Grid item xs = { 6 } >
                <
                Typography variant = "body2"
                color = "textSecondary" >
                Confidence <
                /Typography> <
                Typography variant = "h6" > {
                    (prediction.confidence * 100).toFixed(1) } %
                <
                /Typography> <
                /Grid> <
                Grid item xs = { 6 } >
                <
                Typography variant = "body2"
                color = "textSecondary" >
                Model Used <
                /Typography> <
                Chip label = { prediction.model_used }
                color = "primary"
                size = "small" /
                >
                <
                /Grid> <
                Grid item xs = { 12 } >
                <
                Typography variant = "body2"
                color = "textSecondary" >
                Prediction ID <
                /Typography> <
                Typography variant = "body2" > #{ prediction.id } <
                /Typography> <
                /Grid> <
                /Grid> <
                /Box>
            ) : ( <
                Typography color = "textSecondary" >
                Make a prediction to see results here <
                /Typography>
            )
        } <
        /CardContent> <
        /Card> <
        /Grid>

        { /* Recent Predictions */ } <
        Grid item xs = { 12 } >
        <
        Card >
        <
        CardContent >
        <
        Typography variant = "h6"
        gutterBottom >
        Recent Predictions <
        /Typography>

        <
        TableContainer component = { Paper } >
        <
        Table >
        <
        TableHead >
        <
        TableRow >
        <
        TableCell > Player < /TableCell> <
        TableCell > Prediction < /TableCell> <
        TableCell > Confidence < /TableCell> <
        TableCell > Model < /TableCell> <
        TableCell > Date < /TableCell> <
        /TableRow> <
        /TableHead> <
        TableBody > {
            recentPredictions.map((pred) => ( <
                TableRow key = { pred.id } >
                <
                TableCell >
                Player# { pred.player_id } <
                /TableCell> <
                TableCell >
                <
                Chip label = { pred.prediction ? 'TD' : 'No TD' }
                color = { pred.prediction ? 'success' : 'error' }
                size = "small" /
                >
                <
                /TableCell> <
                TableCell > {
                    (pred.confidence * 100).toFixed(1) } %
                <
                /TableCell> <
                TableCell > { pred.model_used || 'N/A' } <
                /TableCell> <
                TableCell > { new Date(pred.created_at).toLocaleDateString() } <
                /TableCell> <
                /TableRow>
            ))
        } <
        /TableBody> <
        /Table> <
        /TableContainer> <
        /CardContent> <
        /Card> <
        /Grid> <
        /Grid> <
        /Box>
    );
};

export default Predictions;
