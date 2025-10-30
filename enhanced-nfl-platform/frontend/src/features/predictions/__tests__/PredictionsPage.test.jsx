import React from 'react';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import axios from 'axios';
import { renderWithProviders } from '../../../test-utils/renderWithProviders.jsx';
import PredictionsPage from '../PredictionsPage.jsx';

jest.mock('axios');

beforeEach(() => {
  axios.get.mockImplementation((url) => {
    if (url.includes('/players')) {
      return Promise.resolve({
        data: [
          { id: 1, first_name: 'Josh', last_name: 'Allen', position: 'QB' }
        ]
      });
    }
    if (url.includes('/predictions?limit=10')) {
      return Promise.resolve({ data: [] });
    }
    return Promise.resolve({ data: [] });
  });

  axios.post.mockResolvedValue({
    data: { touchdown: true, probability: 0.81, top_features: [] }
  });
});

afterEach(() => {
  axios.get.mockReset();
  axios.post.mockReset();
});

describe('PredictionsPage', () => {
  it('submits a prediction when a player is selected', async () => {
    renderWithProviders(<PredictionsPage />);

    const selectInput = await screen.findByLabelText(/Select Player/i);

    fireEvent.mouseDown(selectInput);
    const option = await screen.findByText(/Josh Allen/);
    fireEvent.click(option);

    const button = await screen.findByRole('button', { name: /Predict Touchdown/i });
    fireEvent.click(button);

    await waitFor(() => expect(axios.post).toHaveBeenCalled());
  });
});
