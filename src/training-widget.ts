import { addFrameUsingScales, drawLine, getContext } from 'web-ui-common/canvas';
import type { Pair } from 'web-ui-common/types';
import { makeScale } from 'web-ui-common/util';

export interface TrainingWidget {
  update: (lossHistory: Pair<number>[]) => void;
  setMaxEpochs: (maxEpochs: number) => void;
  getLossHistory: () => Pair<number>[];
  setLossHistory: (history: Pair<number>[]) => void;
  trainButton: HTMLButtonElement;
  resetButton: HTMLButtonElement;
  statusText: HTMLSpanElement;
}

export function initWidget(container: HTMLDivElement): TrainingWidget {
  // Clear container
  container.innerHTML = '';

  // Create control panel with buttons
  const controlPanel = document.createElement('div');
  controlPanel.style.marginBottom = '10px';

  const trainButton = document.createElement('button');
  trainButton.textContent = 'Train model';
  trainButton.id = 'train-btn';
  controlPanel.appendChild(trainButton);

  const resetButton = document.createElement('button');
  resetButton.textContent = 'Reset model';
  resetButton.id = 'reset-btn';
  resetButton.style.marginLeft = '5px';
  controlPanel.appendChild(resetButton);

  const statusText = document.createElement('span');
  statusText.id = 'train-status';
  statusText.style.marginLeft = '10px';
  statusText.style.color = '#666';
  controlPanel.appendChild(statusText);

  container.appendChild(controlPanel);

  // Create canvas
  const canvas = document.createElement('canvas');
  canvas.width = 600;
  canvas.height = 400;
  container.appendChild(canvas);

  const ctx = getContext(canvas);

  // Track the maximum number of epochs for fixed x-axis
  let maxEpochs = 1000; // Default value

  // Track the current loss history
  let currentLossHistory: Pair<number>[] = [];

  function setMaxEpochs(epochs: number): void {
    maxEpochs = epochs;
  }

  function getLossHistory(): Pair<number>[] {
    return currentLossHistory;
  }

  function setLossHistory(history: Pair<number>[]): void {
    currentLossHistory = history;
    update(history);
  }

  function update(lossHistory: Pair<number>[]): void {
    // Update current history
    currentLossHistory = lossHistory;

    if (lossHistory.length === 0) {
      return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Find data bounds
    const losses = lossHistory.map(([, loss]) => loss);
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);

    // Start with fixed y-axis range, extend only if needed
    const defaultYMin = -0.1;
    const defaultYMax = 0.7;
    let yMin = defaultYMin;
    let yMax = defaultYMax;

    // Extend range if data exceeds default bounds
    if (minLoss < defaultYMin) {
      yMin = minLoss - 0.05;
    }
    if (maxLoss > defaultYMax) {
      yMax = maxLoss + 0.05;
    }

    // Create scales with fixed x-axis (0 to maxEpochs)
    const xScale = makeScale(
      [0, maxEpochs],
      [40, canvas.width - 40]
    );
    const yScale = makeScale(
      [yMin, yMax],
      [canvas.height - 40, 10]
    );

    // Draw frame with axes
    addFrameUsingScales(ctx, xScale, yScale, 6);

    // Draw loss curve
    drawLine(ctx, xScale, yScale, lossHistory, {
      stroke: 'steelblue',
      lineWidth: 2
    });
  }

  return {
    update,
    setMaxEpochs,
    getLossHistory,
    setLossHistory,
    trainButton,
    resetButton,
    statusText
  };
}
