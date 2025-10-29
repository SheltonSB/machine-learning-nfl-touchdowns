import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    Grid,
    TextField,
    Button,
    Chip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
} from '@mui/material';
import {
    Add,
    Search,
    Edit,
    Delete,
    Person,
} from '@mui/icons-material';
import { api } from '../services/api';

const Players = () => {
    const [players, setPlayers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [positionFilter, setPositionFilter] = useState('');
    const [teamFilter, setTeamFilter] = useState('');
    const [openDialog, setOpenDialog] = useState(false);
    const [selectedPlayer, setSelectedPlayer] = useState(null);

    useEffect(() => {
        loadPlayers();
    }, []);

    const loadPlayers = async() => {
        try {
            setLoading(true);
            const response = await api.get('/players');
            setPlayers(response.data);
        } catch (error) {
            console.error('Error loading players:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = () => {
        // Filter players based on search criteria
        // This would typically be done on the backend
        console.log('Searching players:', { searchTerm, positionFilter, teamFilter });
    };

    const handleAddPlayer = () => {
        setSelectedPlayer(null);
        setOpenDialog(true);
    };

    const handleEditPlayer = (player) => {
        setSelectedPlayer(player);
        setOpenDialog(true);
    };

    const handleDeletePlayer = async(playerId) => {
        if (window.confirm('Are you sure you want to delete this player?')) {
            try {
                await api.delete(`/players/${playerId}`);
                loadPlayers();
            } catch (error) {
                console.error('Error deleting player:', error);
            }
        }
    };

    const filteredPlayers = players.filter(player => {
        const matchesSearch = !searchTerm ||
            player.first_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            player.last_name.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesPosition = !positionFilter || player.position === positionFilter;
        const matchesTeam = !teamFilter || player.current_team === teamFilter;

        return matchesSearch && matchesPosition && matchesTeam;
    });

    return ( <
        Box >
        <
        Box display = "flex"
        justifyContent = "space-between"
        alignItems = "center"
        mb = { 3 } >
        <
        Typography variant = "h4"
        component = "h1" >
        Players Management <
        /Typography> <
        Button variant = "contained"
        startIcon = { < Add / > }
        onClick = { handleAddPlayer } >
        Add Player <
        /Button> <
        /Box>

        { /* Search and Filters */ } <
        Card sx = {
            { mb: 3 } } >
        <
        CardContent >
        <
        Grid container spacing = { 2 }
        alignItems = "center" >
        <
        Grid item xs = { 12 }
        md = { 4 } >
        <
        TextField fullWidth label = "Search Players"
        value = { searchTerm }
        onChange = {
            (e) => setSearchTerm(e.target.value) }
        InputProps = {
            {
                startAdornment: < Search sx = {
                    { mr: 1, color: 'text.secondary' } }
                />
            }
        }
        /> <
        /Grid> <
        Grid item xs = { 12 }
        md = { 3 } >
        <
        FormControl fullWidth >
        <
        InputLabel > Position < /InputLabel> <
        Select value = { positionFilter }
        onChange = {
            (e) => setPositionFilter(e.target.value) } >
        <
        MenuItem value = "" > All Positions < /MenuItem> <
        MenuItem value = "QB" > Quarterback < /MenuItem> <
        MenuItem value = "WR" > Wide Receiver < /MenuItem> <
        MenuItem value = "RB" > Running Back < /MenuItem> <
        MenuItem value = "TE" > Tight End < /MenuItem> <
        /Select> <
        /FormControl> <
        /Grid> <
        Grid item xs = { 12 }
        md = { 3 } >
        <
        FormControl fullWidth >
        <
        InputLabel > Team < /InputLabel> <
        Select value = { teamFilter }
        onChange = {
            (e) => setTeamFilter(e.target.value) } >
        <
        MenuItem value = "" > All Teams < /MenuItem> <
        MenuItem value = "KC" > Kansas City Chiefs < /MenuItem> <
        MenuItem value = "TB" > Tampa Bay Buccaneers < /MenuItem> <
        MenuItem value = "GB" > Green Bay Packers < /MenuItem> <
        MenuItem value = "BUF" > Buffalo Bills < /MenuItem> <
        /Select> <
        /FormControl> <
        /Grid> <
        Grid item xs = { 12 }
        md = { 2 } >
        <
        Button fullWidth variant = "outlined"
        onClick = { handleSearch } >
        Search <
        /Button> <
        /Grid> <
        /Grid> <
        /CardContent> <
        /Card>

        { /* Players Table */ } <
        Card >
        <
        TableContainer >
        <
        Table >
        <
        TableHead >
        <
        TableRow >
        <
        TableCell > Player < /TableCell> <
        TableCell > Position < /TableCell> <
        TableCell > Team < /TableCell> <
        TableCell > Age < /TableCell> <
        TableCell > Experience < /TableCell> <
        TableCell > Actions < /TableCell> <
        /TableRow> <
        /TableHead> <
        TableBody > {
            filteredPlayers.map((player) => ( <
                TableRow key = { player.id } >
                <
                TableCell >
                <
                Box display = "flex"
                alignItems = "center" >
                <
                Person sx = {
                    { mr: 1, color: 'text.secondary' } }
                /> <
                Box >
                <
                Typography variant = "subtitle2" > { player.first_name } { player.last_name } <
                /Typography> <
                Typography variant = "caption"
                color = "text.secondary" >
                ID: { player.player_id } <
                /Typography> <
                /Box> <
                /Box> <
                /TableCell> <
                TableCell >
                <
                Chip label = { player.position }
                color = "primary"
                size = "small" /
                >
                <
                /TableCell> <
                TableCell > { player.current_team || 'N/A' } < /TableCell> <
                TableCell > { player.age || 'N/A' } < /TableCell> <
                TableCell > { player.experience || 'N/A' }
                years < /TableCell> <
                TableCell >
                <
                IconButton size = "small"
                onClick = {
                    () => handleEditPlayer(player) } >
                <
                Edit / >
                <
                /IconButton> <
                IconButton size = "small"
                onClick = {
                    () => handleDeletePlayer(player.id) }
                color = "error" >
                <
                Delete / >
                <
                /IconButton> <
                /TableCell> <
                /TableRow>
            ))
        } <
        /TableBody> <
        /Table> <
        /TableContainer> <
        /Card>

        { /* Add/Edit Player Dialog */ } <
        Dialog open = { openDialog }
        onClose = {
            () => setOpenDialog(false) }
        maxWidth = "md"
        fullWidth >
        <
        DialogTitle > { selectedPlayer ? 'Edit Player' : 'Add New Player' } <
        /DialogTitle> <
        DialogContent >
        <
        Typography >
        Player form would go here.This is a placeholder
        for the actual form. <
        /Typography> <
        /DialogContent> <
        DialogActions >
        <
        Button onClick = {
            () => setOpenDialog(false) } > Cancel < /Button> <
        Button variant = "contained" > { selectedPlayer ? 'Update' : 'Add' }
        Player <
        /Button> <
        /DialogActions> <
        /Dialog> <
        /Box>
    );
};

export default Players;
