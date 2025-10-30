import React from 'react';
import PropTypes from 'prop-types';
import { Alert } from '@mui/material';

function ErrorState({ message }) {
    return (
        <Alert severity="error" sx={{ mt: 2 }}>
            {message}
        </Alert>
    );
}

ErrorState.propTypes = {
    message: PropTypes.string.isRequired,
};

export default ErrorState;
