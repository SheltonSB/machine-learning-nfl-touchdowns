import React from 'react';
import {
    AppBar,
    Toolbar,
    Typography,
    Button,
    Box,
    IconButton,
    Menu,
    MenuItem,
} from '@mui/material';
import {
    SportsFootball,
    Dashboard,
    People,
    Analytics,
    Chat,
    Menu as MenuIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Navbar = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [anchorEl, setAnchorEl] = React.useState(null);

    const handleMenuOpen = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
    };

    const handleNavigation = (path) => {
        navigate(path);
        handleMenuClose();
    };

    const menuItems = [
        { path: '/', label: 'Dashboard', icon: < Dashboard / > },
        { path: '/players', label: 'Players', icon: < People / > },
        { path: '/predictions', label: 'Predictions', icon: < SportsFootball / > },
        { path: '/analytics', label: 'Analytics', icon: < Analytics / > },
        { path: '/chat', label: 'AI Chat', icon: < Chat / > },
    ];

    return ( <
        AppBar position = "static"
        elevation = { 2 } >
        <
        Toolbar >
        <
        IconButton size = "large"
        edge = "start"
        color = "inherit"
        aria - label = "menu"
        sx = {
            { mr: 2 } }
        onClick = { handleMenuOpen } >
        <
        MenuIcon / >
        <
        /IconButton>

        <
        Typography variant = "h6"
        component = "div"
        sx = {
            { flexGrow: 1, display: 'flex', alignItems: 'center' } } >
        <
        SportsFootball sx = {
            { mr: 1 } }
        />
        NFL AI Platform <
        /Typography>

        <
        Box sx = {
            { display: { xs: 'none', md: 'flex' }, gap: 1 } } > {
            menuItems.map((item) => ( <
                Button key = { item.path }
                color = "inherit"
                startIcon = { item.icon }
                onClick = {
                    () => handleNavigation(item.path) }
                sx = {
                    {
                        backgroundColor: location.pathname === item.path ? 'rgba(255,255,255,0.1)' : 'transparent',
                    }
                } >
                { item.label } <
                /Button>
            ))
        } <
        /Box>

        <
        Menu anchorEl = { anchorEl }
        open = { Boolean(anchorEl) }
        onClose = { handleMenuClose }
        sx = {
            { display: { xs: 'block', md: 'none' } } } >
        {
            menuItems.map((item) => ( <
                MenuItem key = { item.path }
                onClick = {
                    () => handleNavigation(item.path) }
                selected = { location.pathname === item.path } >
                { item.icon } <
                Typography sx = {
                    { ml: 1 } } > { item.label } < /Typography> <
                /MenuItem>
            ))
        } <
        /Menu> <
        /Toolbar> <
        /AppBar>
    );
};

export default Navbar;
