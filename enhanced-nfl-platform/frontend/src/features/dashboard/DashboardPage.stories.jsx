import React from 'react';
import DashboardPage from './DashboardPage.jsx';
import { rest } from 'msw';
import { within } from '@storybook/testing-library';

export default {
  title: 'Features/DashboardPage',
  component: DashboardPage,
  parameters: {
    layout: 'fullscreen',
    msw: {
      handlers: [
        rest.get('/api/v1/analytics/overview', (req, res, ctx) =>
          res(
            ctx.json({
              total_players: 150,
              total_predictions: 3200,
              accuracy: 0.88,
              active_models: 4
            })
          )
        ),
        rest.get('/api/v1/predictions', (req, res, ctx) =>
          res(
            ctx.json([
              {
                id: 1,
                player_id: 10,
                player_name: 'Patrick Mahomes',
                opponent: 'BUF',
                prediction: true,
                confidence: 0.91
              }
            ])
          )
        ),
        rest.get('/api/v1/predictions/model/performance', (req, res, ctx) =>
          res(
            ctx.json({
              xgboost: { accuracy: 0.91, f1_score: 0.88 },
              tensorflow: { accuracy: 0.9, f1_score: 0.87 }
            })
          )
        )
      ]
    }
  }
};

const Template = () => <DashboardPage />;

export const Default = Template.bind({});

Default.play = async ({ canvasElement }) => {
  const canvas = within(canvasElement);
  await canvas.findByText(/dashboard/i);
};
