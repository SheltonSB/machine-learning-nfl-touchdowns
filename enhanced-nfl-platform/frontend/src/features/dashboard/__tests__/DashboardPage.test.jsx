import React from 'react';
import { waitFor, screen } from '@testing-library/react';
import axios from 'axios';
import { renderWithProviders } from '../../../test-utils/renderWithProviders.jsx';
import DashboardPage from '../DashboardPage.jsx';

jest.mock('axios');

beforeEach(() => {
  axios.get.mockImplementation((url) => {
    if (url.includes('/analytics/overview')) {
      return Promise.resolve({ data: { total_players: 100, total_predictions: 2500, accuracy: 0.9, active_models: 3 } });
    }
    if (url.includes('/predictions?limit=5')) {
      return Promise.resolve({ data: [] });
    }
    if (url.includes('/predictions/model/performance')) {
      return Promise.resolve({ data: { ensemble: { accuracy: 0.93, f1_score: 0.9 } } });
    }
    if (url.includes('/analytics/players')) {
      return Promise.resolve({ data: { position_distribution: {} } });
    }
    if (url.includes('/analytics/teams')) {
      return Promise.resolve({ data: { touchdown_distribution: {} } });
    }
    if (url.includes('/analytics/trends')) {
      return Promise.resolve({ data: { weekly_touchdowns: [] } });
    }
    return Promise.resolve({ data: [] });
  });
});

afterEach(() => {
  axios.get.mockReset();
});

describe('DashboardPage', () => {
  it('renders heading and metrics', async () => {
    renderWithProviders(<DashboardPage />);

    expect(screen.getByText(/Loading dashboard/i)).toBeInTheDocument();

    await waitFor(() => expect(screen.getByText(/NFL AI Platform Dashboard/i)).toBeInTheDocument());
    expect(screen.getByText(/Total Players/i)).toBeInTheDocument();
  });
});
