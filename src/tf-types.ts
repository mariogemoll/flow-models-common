import type * as tfjs from '@tensorflow/tfjs';

// Type definitions for the global tf object
declare global {
  const tf: typeof tfjs;
}

// Export commonly used types
export type Tensor = tfjs.Tensor;
export type Tensor1D = tfjs.Tensor1D;
export type Tensor2D = tfjs.Tensor2D;
export type LayerVariable = tfjs.LayerVariable;
export type Variable = tfjs.Variable;

export {};
