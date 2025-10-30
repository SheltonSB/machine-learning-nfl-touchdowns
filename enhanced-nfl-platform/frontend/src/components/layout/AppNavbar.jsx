import React, { useState } from 'react';
import PropTypes from 'prop-types';
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
    SportsFootball as SportsFootballIcon,
    Dashboard as DashboardIcon,
    People as PeopleIcon,
    Analytics as AnalyticsIcon,
    Chat as ChatIcon,
    SportsScore as SportsScoreIcon,
    Menu as MenuIcon,
} from '@mui/icons-material';
import { useLocation, useNavigate } from 'react-router-dom';

const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon fontSize="small" /> },
    { path: '/players', label: 'Players', icon: <PeopleIcon fontSize="small" /> },
    { path: '/predictions', label: 'Predictions', icon: <SportsScoreIcon fontSize="small" /> },
    { path: '/analytics', label: 'Analytics', icon: <AnalyticsIcon fontSize="small" /> },
    { path: '/chat', label: 'AI Chat', icon: <ChatIcon fontSize="small" /> },
];

function AppNavbar({ title }) {
    const location = useLocation();
    const navigate = useNavigate();
    const [anchorEl, setAnchorEl] = useState(null);

    const handleNavigate = (path) => {
        navigate(path);
        setAnchorEl(null);
    };

    return (
        <AppBar position="static" elevation={2}>
            <Toolbar>
                <IconButton
                    size="large"
                    edge="start"
                    color="inherit"
                    aria-label="open navigation"
                    sx={{ mr: 2, display: { md: 'none' } }}
                    onClick={(event) => setAnchorEl(event.currentTarget)}
                >
                    <MenuIcon />
                </IconButton>

                <Typography
                    variant="h6"
                    component="div"
                    sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}
                >
                    <SportsFootballIcon sx={{ mr: 1 }} />
                    {title}
                </Typography>

                <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 1 }}>
                    {navItems.map((item) => (
                        <Button
                            key={item.path}
                            color="inherit"
                            startIcon={item.icon}
                            onClick={() => handleNavigate(item.path)}
                            sx={{
                                bgcolor:
                                    location.pathname === item.path
                                        ? 'rgba(255,255,255,0.12)'
                                        : 'transparent',
                            }}
                        >
                            {item.label}
                        </Button>
                    ))}
                </Box>

                <Menu
                    anchorEl={anchorEl}
                    open={Boolean(anchorEl)}
                    onClose={() => setAnchorEl(null)}
                    sx={{ display: { xs: 'block', md: 'none' } }}
                >
                    {navItems.map((item) => (
                        <MenuItem
                            key={item.path}
                            selected={location.pathname === item.path}
                            onClick={() => handleNavigate(item.path)}
                        >
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {item.icon}
                                <Typography>{item.label}</Typography>
                            </Box>
                        </MenuItem>
                    ))}
                </Menu>
            </Toolbar>
        </AppBar>
    );
}

AppNavbar.propTypes = {
    title: PropTypes.string,
};

AppNavbar.defaultProps = {
    title: 'NFL AI Platform',
};

export default AppNavbar;
