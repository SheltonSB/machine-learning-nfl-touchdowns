import React from 'react';
import PropTypes from 'prop-types';
import { Box, LinearProgress, Typography } from '@mui/material';

function LoadingState({ message }) {
    return (
        <Box sx={{ width: '100%', mt: 2 }}>
            <LinearProgress />
            {message && (
                <Typography variant="body2" sx={{ mt: 2, textAlign: 'center' }}>
                    {message}
                </Typography>
            )}
        </Box>
    );
}

LoadingState.propTypes = {
    message: PropTypes.string,
};

LoadingState.defaultProps = {
    message: 'Loading...',
};

export default LoadingState;
