import type * as tfjs from '@tensorflow/tfjs';
import type { LayerVariable, Tensor1D, Tensor2D } from './tf-types';

/**
 * Base interface for all flow-based models that transform p_init -> p_data
 * This includes normalizing flows, flow matching, and diffusion models
 */
export interface FlowModel {
  /**
   * Generate samples from initial distribution (p_init) to data distribution (p_data)
   * Returns array of intermediate steps (for visualization) and log probability if applicable
   *
   * For normalizing flows: inverse transform (latent -> data)
   * For flow matching: solve ODE backwards (noise -> data)
   * For diffusion: reverse diffusion process (noise -> data)
   */
  generate(z: Tensor2D): [Tensor2D[], Tensor1D | null];

  /**
   * Compute the loss for training
   */
  computeLoss(x: Tensor2D): tfjs.Scalar;

  /**
   * Get trainable weights for optimization
   */
  getTrainableWeights(): LayerVariable[];

  /**
   * Load model weights from a URL or path
   */
  loadWeights(modelPath: string): Promise<boolean>;

  /**
   * Save model weights (typically to downloads)
   */
  saveWeights(): Promise<void>;
}

/**
 * Factory function type for creating model instances
 */
export type ModelFactory<T extends FlowModel = FlowModel> = () => T;
