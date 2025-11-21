import type { FlowModel } from './model-interface';
import type { Tensor2D } from './tf-types';

export type TrainingState = 'not_started' | 'training' | 'paused' | 'completed';

/**
 * Pipeline state object
 */
export interface PipelineState {
  numEpochs: number;
  trainData: Tensor2D | null;
  model: FlowModel | null;
  trainingState: TrainingState;
}

// Legacy alias for backwards compatibility
export type PageState = PipelineState;

/**
 * Interface for widgets that need model access
 */
export interface ModelState {
  model: FlowModel | null;
}

/**
 * Interface for widgets that need training data access
 */
export interface TrainDataState {
  trainData: Tensor2D | null;
}
