import React, { useEffect, useMemo, useState } from 'react';
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    FormControl,
    Grid,
    InputLabel,
    MenuItem,
    Select,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField,
    Typography,
    Paper,
} from '@mui/material';
import {
    SportsFootball as SportsFootballIcon,
    TrendingUp as TrendingUpIcon,
    Psychology as PsychologyIcon,
} from '@mui/icons-material';

import { PageHeader, LoadingState, ErrorState } from '../../components/common';
import { api } from '../../services/api';

const defaultFeatures = {
    passing_yards_roll3: 250,
    td_passes_roll3: 1.5,
    passes_attempted_roll3: 35,
    age: 28,
    experience: 5,
    height: 74,
    weight: 220,
};

const featureInputs = [
    { key: 'passing_yards_roll3', label: 'Passing Yards (avg)', step: 1 },
    { key: 'td_passes_roll3', label: 'TD Passes (avg)', step: 0.1 },
    { key: 'passes_attempted_roll3', label: 'Pass Attempts (avg)', step: 1 },
    { key: 'age', label: 'Age', step: 1 },
    { key: 'experience', label: 'Experience (years)', step: 1 },
    { key: 'height', label: 'Height (inches)', step: 1 },
    { key: 'weight', label: 'Weight (lbs)', step: 1 },
];

const modelOptions = [
    { value: 'ensemble', label: 'Ensemble (recommended)' },
    { value: 'xgboost', label: 'XGBoost' },
    { value: 'tensorflow', label: 'TensorFlow' },
    { value: 'pytorch', label: 'PyTorch' },
];

function PredictionsPage() {
    const [players, setPlayers] = useState([]);
    const [recentPredictions, setRecentPredictions] = useState([]);
    const [selectedPlayer, setSelectedPlayer] = useState('');
    const [modelName, setModelName] = useState('ensemble');
    const [features, setFeatures] = useState(defaultFeatures);
    const [prediction, setPrediction] = useState(null);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [submitting, setSubmitting] = useState(false);

    useEffect(() => {
        const fetchInitialData = async () => {
            try {
                setLoading(true);
                const [playersResponse, predictionsResponse] = await Promise.all([
                    api.get('/players?limit=50'),
                    api.get('/predictions?limit=10'),
                ]);
                setPlayers(playersResponse.data);
                setRecentPredictions(predictionsResponse.data);
            } catch (err) {
                console.error('Failed to load prediction data:', err);
                setError('Unable to load prediction data. Please try again later.');
            } finally {
                setLoading(false);
            }
        };

        fetchInitialData();
    }, []);

    const handleFeatureChange = (key, value) => {
        setFeatures((prev) => ({
            ...prev,
            [key]: parseFloat(value) || 0,
        }));
    };

    const playerOptions = useMemo(() => {
        return players.map((player) => ({
            value: player.id,
            label: `${player.first_name} ${player.last_name} (${player.position})`,
        }));
    }, [players]);

    const submitPrediction = async () => {
        if (!selectedPlayer) {
            setError('Please select a player before making a prediction.');
            return;
        }

        try {
            setSubmitting(true);
            setError(null);
            const response = await api.post('/predictions', {
                player_id: Number(selectedPlayer),
                features,
                model_name: modelName,
            });
            setPrediction(response.data);
            const history = await api.get('/predictions?limit=10');
            setRecentPredictions(history.data);
        } catch (err) {
            console.error('Error making prediction:', err);
            setError('Unable to make prediction. Please try again.');
        } finally {
            setSubmitting(false);
        }
    };

    if (loading) {
        return <LoadingState message="Loading prediction tools..." />;
    }

    if (error && !submitting) {
        return <ErrorState message={error} />;
    }

    return (
        <Box>
            <PageHeader
                title="Touchdown Predictions"
                subtitle="Run the ensemble or individual models against recent player data."
            />

            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <SportsFootballIcon fontSize="small" />
                                Make Prediction
                            </Typography>

                            <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <FormControl fullWidth>
                                        <InputLabel id="player-select-label">Select Player</InputLabel>
                                        <Select
                                            labelId="player-select-label"
                                            value={selectedPlayer}
                                            label="Select Player"
                                            onChange={(event) => setSelectedPlayer(event.target.value)}
                                        >
                                            {playerOptions.map((option) => (
                                                <MenuItem key={option.value} value={option.value}>
                                                    {option.label}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </Grid>

                                <Grid item xs={12}>
                                    <FormControl fullWidth>
                                        <InputLabel id="model-select-label">ML Model</InputLabel>
                                        <Select
                                            labelId="model-select-label"
                                            value={modelName}
                                            label="ML Model"
                                            onChange={(event) => setModelName(event.target.value)}
                                        >
                                            {modelOptions.map((option) => (
                                                <MenuItem key={option.value} value={option.value}>
                                                    {option.label}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </Grid>

                                {featureInputs.map(({ key, label, step }) => (
                                    <Grid item xs={12} sm={6} key={key}>
                                        <TextField
                                            fullWidth
                                            type="number"
                                            label={label}
                                            value={features[key]}
                                            inputProps={{ step }}
                                            onChange={(event) => handleFeatureChange(key, event.target.value)}
                                        />
                                    </Grid>
                                ))}

                                <Grid item xs={12}>
                                    <Button
                                        variant="contained"
                                        fullWidth
                                        disabled={submitting}
                                        onClick={submitPrediction}
                                    >
                                        {submitting ? 'Running Prediction...' : 'Predict Touchdown'}
                                    </Button>
                                </Grid>
                            </Grid>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <Typography
                                variant="h6"
                                gutterBottom
                                sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
                            >
                                <PsychologyIcon fontSize="small" />
                                Prediction Output
                            </Typography>

                            {error && submitting && <Alert severity="error">{error}</Alert>}

                            {prediction ? (
                                <Box>
                                    <Typography variant="h5" gutterBottom>
                                        {prediction.touchdown ? 'Touchdown Expected' : 'No Touchdown Expected'}
                                    </Typography>
                                    <Typography variant="body1" gutterBottom>
                                        Probability:{' '}
                                        <strong>{Math.round(prediction.probability * 100)}%</strong>
                                    </Typography>
                                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
                                        {prediction.top_features?.map((feature) => (
                                            <Chip
                                                key={feature.name}
                                                label={`${feature.name}: ${feature.value}`}
                                                color="primary"
                                                variant="outlined"
                                            />
                                        ))}
                                    </Box>
                                </Box>
                            ) : (
                                <Typography color="text.secondary">
                                    Make a prediction to view the model output and key driver features.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>

                    <Card sx={{ mt: 3 }}>
                        <CardContent>
                            <Typography
                                variant="h6"
                                gutterBottom
                                sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
                            >
                                <TrendingUpIcon fontSize="small" />
                                Recent Predictions
                            </Typography>

                            <TableContainer component={Paper}>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Player</TableCell>
                                            <TableCell>Opponent</TableCell>
                                            <TableCell>Model</TableCell>
                                            <TableCell>Probability</TableCell>
                                            <TableCell>Result</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {recentPredictions.map((item) => (
                                            <TableRow key={item.id || item.prediction_id}>
                                                <TableCell>{item.player_name}</TableCell>
                                                <TableCell>{item.opponent}</TableCell>
                                                <TableCell>{item.model_name || 'ensemble'}</TableCell>
                                                <TableCell>
                                                    {item.confidence != null
                                                        ? `${Math.round(item.confidence * 100)}%`
                                                        : '—'}
                                                </TableCell>
                                                <TableCell>
                                                    <Chip
                                                        label={item.prediction ? 'Touchdown' : 'No TD'}
                                                        color={item.prediction ? 'success' : 'default'}
                                                        size="small"
                                                    />
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                        {!recentPredictions.length && (
                                            <TableRow>
                                                <TableCell colSpan={5} align="center">
                                                    <Typography color="text.secondary">
                                                        No recent predictions available.
                                                    </Typography>
                                                </TableCell>
                                            </TableRow>
                                        )}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
}

export default PredictionsPage;
