import React, { useEffect, useMemo, useState } from 'react';
import {
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    FormControl,
    Grid,
    IconButton,
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
    Add as AddIcon,
    Delete as DeleteIcon,
    Edit as EditIcon,
    Person as PersonIcon,
    Search as SearchIcon,
} from '@mui/icons-material';

import { PageHeader, LoadingState, ErrorState } from '../../components/common';
import { api } from '../../services/api';

const positionOptions = [
    { value: '', label: 'All Positions' },
    { value: 'QB', label: 'Quarterback' },
    { value: 'WR', label: 'Wide Receiver' },
    { value: 'RB', label: 'Running Back' },
    { value: 'TE', label: 'Tight End' },
];

const teamOptions = [
    { value: '', label: 'All Teams' },
    { value: 'KC', label: 'Kansas City Chiefs' },
    { value: 'TB', label: 'Tampa Bay Buccaneers' },
    { value: 'GB', label: 'Green Bay Packers' },
    { value: 'BUF', label: 'Buffalo Bills' },
];

function PlayersPage() {
    const [players, setPlayers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [searchTerm, setSearchTerm] = useState('');
    const [positionFilter, setPositionFilter] = useState('');
    const [teamFilter, setTeamFilter] = useState('');

    const [dialogOpen, setDialogOpen] = useState(false);
    const [selectedPlayer, setSelectedPlayer] = useState(null);

    useEffect(() => {
        const fetchPlayers = async () => {
            try {
                setLoading(true);
                const response = await api.get('/players');
                setPlayers(response.data);
            } catch (err) {
                console.error('Error loading players:', err);
                setError('Unable to load players. Please try again later.');
            } finally {
                setLoading(false);
            }
        };

        fetchPlayers();
    }, []);

    const filteredPlayers = useMemo(() => {
        return players.filter((player) => {
            const matchesSearch =
                !searchTerm ||
                player.first_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                player.last_name.toLowerCase().includes(searchTerm.toLowerCase());
            const matchesPosition = !positionFilter || player.position === positionFilter;
            const matchesTeam = !teamFilter || player.current_team === teamFilter;
            return matchesSearch && matchesPosition && matchesTeam;
        });
    }, [players, searchTerm, positionFilter, teamFilter]);

    const openPlayerDialog = (player = null) => {
        setSelectedPlayer(player);
        setDialogOpen(true);
    };

    const handleDeletePlayer = async (playerId) => {
        if (!window.confirm('Are you sure you want to delete this player?')) {
            return;
        }
        try {
            await api.delete(`/players/${playerId}`);
            setPlayers((prev) => prev.filter((player) => player.id !== playerId));
        } catch (err) {
            console.error('Error deleting player:', err);
            setError('Unable to delete player. Please try again.');
        }
    };

    if (loading) {
        return <LoadingState message="Loading players..." />;
    }

    if (error) {
        return <ErrorState message={error} />;
    }

    return (
        <Box>
            <PageHeader
                title="Players Management"
                subtitle="Search, review, and curate quarterback data used by the prediction models"
                action={
                    <Button variant="contained" startIcon={<AddIcon />} onClick={() => openPlayerDialog()}>
                        Add Player
                    </Button>
                }
            />

            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                label="Search Players"
                                value={searchTerm}
                                onChange={(event) => setSearchTerm(event.target.value)}
                                InputProps={{
                                    startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                                }}
                            />
                        </Grid>
                        <Grid item xs={12} md={3}>
                            <FormControl fullWidth>
                                <InputLabel id="position-filter-label">Position</InputLabel>
                                <Select
                                    labelId="position-filter-label"
                                    value={positionFilter}
                                    label="Position"
                                    onChange={(event) => setPositionFilter(event.target.value)}
                                >
                                    {positionOptions.map((option) => (
                                        <MenuItem key={option.value} value={option.value}>
                                            {option.label}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} md={3}>
                            <FormControl fullWidth>
                                <InputLabel id="team-filter-label">Team</InputLabel>
                                <Select
                                    labelId="team-filter-label"
                                    value={teamFilter}
                                    label="Team"
                                    onChange={(event) => setTeamFilter(event.target.value)}
                                >
                                    {teamOptions.map((option) => (
                                        <MenuItem key={option.value} value={option.value}>
                                            {option.label}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} md={2}>
                            <Button
                                fullWidth
                                variant="outlined"
                                onClick={() => {
                                    console.info('Filter applied', {
                                        search: searchTerm,
                                        position: positionFilter,
                                        team: teamFilter,
                                    });
                                }}
                            >
                                Search
                            </Button>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>

            <Card>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Player</TableCell>
                                <TableCell>Position</TableCell>
                                <TableCell>Team</TableCell>
                                <TableCell>Age</TableCell>
                                <TableCell>Experience</TableCell>
                                <TableCell align="right">Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredPlayers.map((player) => (
                                <TableRow key={player.id}>
                                    <TableCell>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <PersonIcon sx={{ color: 'text.secondary' }} />
                                            <Box>
                                                <Typography variant="subtitle2">
                                                    {player.first_name} {player.last_name}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    ID: {player.player_id}
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </TableCell>
                                    <TableCell>
                                        <Chip label={player.position} color="primary" size="small" />
                                    </TableCell>
                                    <TableCell>{player.current_team || 'N/A'}</TableCell>
                                    <TableCell>{player.age || 'N/A'}</TableCell>
                                    <TableCell>
                                        {player.experience != null ? `${player.experience} years` : 'N/A'}
                                    </TableCell>
                                    <TableCell align="right">
                                        <IconButton size="small" onClick={() => openPlayerDialog(player)}>
                                            <EditIcon fontSize="small" />
                                        </IconButton>
                                        <IconButton
                                            size="small"
                                            color="error"
                                            onClick={() => handleDeletePlayer(player.id)}
                                        >
                                            <DeleteIcon fontSize="small" />
                                        </IconButton>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {!filteredPlayers.length && (
                                <TableRow>
                                    <TableCell colSpan={6} align="center">
                                        <Typography color="text.secondary">
                                            No players match the current filters.
                                        </Typography>
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Card>

            <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
                <DialogTitle>{selectedPlayer ? 'Edit Player' : 'Add New Player'}</DialogTitle>
                <DialogContent>
                    <Typography color="text.secondary">
                        A full player form can be added here. For now this dialog demonstrates how the UI behaves.
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
                    <Button variant="contained" onClick={() => setDialogOpen(false)}>
                        {selectedPlayer ? 'Update Player' : 'Add Player'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}

export default PlayersPage;
