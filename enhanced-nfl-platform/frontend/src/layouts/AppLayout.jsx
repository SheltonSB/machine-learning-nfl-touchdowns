import React from 'react';
import PropTypes from 'prop-types';
import { Box, Container } from '@mui/material';
import { AppNavbar } from '../components/layout';

function AppLayout({ children }) {
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <AppNavbar />
            <Container component="main" sx={{ flexGrow: 1, py: 3 }}>
                {children}
            </Container>
        </Box>
    );
}

AppLayout.propTypes = {
    children: PropTypes.node.isRequired,
};

export default AppLayout;
