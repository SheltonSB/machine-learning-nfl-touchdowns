import React from 'react';
import PropTypes from 'prop-types';
import { Box, Typography } from '@mui/material';

function PageHeader({ title, subtitle, action }) {
    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: { xs: 'column', sm: 'row' },
                alignItems: { xs: 'flex-start', sm: 'center' },
                justifyContent: 'space-between',
                gap: 2,
                mb: 3,
            }}
        >
            <Box>
                <Typography variant="h4" component="h1">
                    {title}
                </Typography>
                {subtitle && (
                    <Typography variant="body2" color="text.secondary">
                        {subtitle}
                    </Typography>
                )}
            </Box>
            {action}
        </Box>
    );
}

PageHeader.propTypes = {
    title: PropTypes.string.isRequired,
    subtitle: PropTypes.string,
    action: PropTypes.node,
};

PageHeader.defaultProps = {
    subtitle: undefined,
    action: null,
};

export default PageHeader;
