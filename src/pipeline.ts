import { loadLossHistory, saveLossHistory } from './loss-history';
import type { FlowModel, Generative, ModelFactory } from './model-interface';
import { initWidget as initMoonsDataset } from './moons-widget';
import { trainModel } from './train';
import { initWidget as initTraining } from './training-widget';
import type { PipelineState, TrainingState } from './types';

/**
 * Create initial pipeline state
 */
export function createPipelineState(numEpochs = 1000): PipelineState {
  return {
    numEpochs,
    trainData: null,
    model: null,
    trainingState: 'not_started'
  };
}

// Legacy alias for backwards compatibility
export const createPageState = createPipelineState;

export interface VisualizationCallbacks {
  updateVisualization: (model: Generative, container: HTMLDivElement) => void;
  showTrainingInProgress?: (container: HTMLDivElement) => void;
}

export async function initPipeline(
  moonsDatasetContainer: HTMLDivElement,
  trainingContainer: HTMLDivElement,
  flowVisualizationContainer: HTMLDivElement,
  modelFactory: ModelFactory<FlowModel & Generative>,
  modelUrl: string,
  lossHistoryUrl: string,
  visualizationCallbacks: VisualizationCallbacks,
  numEpochs = 1000
): Promise<void> {
  // Wait for TensorFlow to be ready before doing anything
  await tf.ready();

  // Create pipeline state
  const state = createPipelineState(numEpochs);

  // Default implementation for showTrainingInProgress if not provided
  const showTrainingInProgress = visualizationCallbacks.showTrainingInProgress ?? (
    (container: HTMLDivElement): void => {
      container.innerHTML = `
        <div style="
          display: flex;
          align-items: center;
          justify-content: center;
          height: 400px;
          background: #f5f5f5;
          border-radius: 4px;
        ">
          <div style="text-align: center; color: #666;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">
              Training in Progress
            </div>
            <div style="font-size: 14px;">
              Visualization will be available when training is paused or completed
            </div>
          </div>
        </div>
      `;
    }
  );

  // Moons dataset widget
  initMoonsDataset(moonsDatasetContainer, state);

  // Training widget
  const trainingWidget = initTraining(trainingContainer);
  trainingWidget.setMaxEpochs(state.numEpochs);

  // Get button and status references from the widget
  const trainButton = trainingWidget.trainButton;
  const resetButton = trainingWidget.resetButton;
  const trainStatus = trainingWidget.statusText;

  // Create model using factory function
  // Track with proper type for visualization callbacks
  let model: FlowModel & Generative = modelFactory();
  state.model = model;
  console.log('Created model using factory function');

  try {
    const success = await state.model.loadWeights(modelUrl);
    if (success) {
      console.log('Loaded weights from model.json');
      trainStatus.textContent = 'Loaded pre-trained weights';
      state.trainingState = 'completed';

      // Try to load loss history
      const lossHistory = await loadLossHistory(lossHistoryUrl);
      if (lossHistory) {
        trainingWidget.setLossHistory(lossHistory);
      }

      // Generate and show visualization
      visualizationCallbacks.updateVisualization(model, flowVisualizationContainer);
    } else {
      trainStatus.textContent = 'Failed to load weights';
    }
  } catch (error) {
    console.log('Could not load model.json:', error);
    trainStatus.textContent =
      'No pre-trained weights found. Click "Train model" to train.';
  }

  // Update button states based on training status
  function updateButtonStates(): void {
    switch (state.trainingState) {
    case 'training':
      trainButton.textContent = 'Pause training';
      trainButton.disabled = false;
      resetButton.disabled = true;
      break;
    case 'paused':
      trainButton.textContent = 'Resume training';
      trainButton.disabled = false;
      resetButton.disabled = false;
      break;
    case 'completed':
      trainButton.textContent = 'Training completed';
      trainButton.disabled = true;
      resetButton.disabled = false;
      break;
    case 'not_started':
      trainButton.textContent = 'Train model';
      trainButton.disabled = false;
      resetButton.disabled = false;
      break;
    }
  }

  // Initial button state
  updateButtonStates();

  // Reset button handler
  resetButton.addEventListener('click', () => {
    // Create new untrained model using factory
    model = modelFactory();
    state.model = model;
    state.trainingState = 'not_started';
    console.log('Reset: Created new untrained model');

    // Clear loss history
    trainingWidget.setLossHistory([]);

    // Update status and visualization
    trainStatus.textContent = 'Model reset. Ready to train.';
    visualizationCallbacks.updateVisualization(model, flowVisualizationContainer);
    updateButtonStates();
  });

  // Train/Pause button handler
  // eslint-disable-next-line @typescript-eslint/no-misused-promises
  trainButton.addEventListener('click', async() => {
    if (state.trainingState === 'training') {
      // Pause training (will be handled by training loop)
      state.trainingState = 'paused';
      trainStatus.textContent = 'Pausing training...';
    } else if (state.trainingState !== 'completed') {
      // Start or resume training (only if not completed)
      await startTraining();
    }
  });

  async function startTraining(): Promise<void> {
    state.trainingState = 'training';
    updateButtonStates();

    // Show training in progress view
    showTrainingInProgress(flowVisualizationContainer);

    trainStatus.textContent = 'Training...';

    // Train and update model in state
    model = await trainModel(state, modelFactory, trainingWidget);
    state.model = model;

    // Store final state before calling other functions (widen type to avoid narrowing issues)
    const finalState = state.trainingState as TrainingState;

    updateButtonStates();

    // Update status based on final state (trainModel may have changed it to 'paused')
    switch (finalState) {
    case 'paused':
      trainStatus.textContent = 'Training paused';
      break;
    case 'completed':
      trainStatus.textContent = 'Training complete!';
      break;
    default:
      // Should not happen, but handle gracefully
      trainStatus.textContent = 'Training finished';
    }

    // Update visualization
    visualizationCallbacks.updateVisualization(model, flowVisualizationContainer);
  }

  // Expose state and utilities globally for console access
  interface WindowWithState {
    state: typeof state;
    saveLossHistory: typeof saveLossHistory;
  }
  const windowWithState = window as unknown as WindowWithState;
  windowWithState.state = state;
  windowWithState.saveLossHistory = (): void => {
    saveLossHistory(trainingWidget.getLossHistory());
  };

  console.log('Page state available as window.state');
  console.log('To save model weights: await state.model.saveWeights()');
  console.log('To save loss history: saveLossHistory()');
}
