import type * as tfjs from '@tensorflow/tfjs';
import type { Pair } from 'web-ui-common/types';

import type { FlowModel, ModelFactory } from './model-interface';
import { makeMoons } from './moons-dataset';
import type { Tensor1D, Tensor2D } from './tf-types';
import type { TrainingWidget } from './training-widget';
import type { PipelineState } from './types';

/**
 * Type guard to check if a model has forward/inverse methods
 * (normalizing flows, flow matching, CNFs)
 */
function hasForwardInverse(model: FlowModel): model is FlowModel & {
  forward: (x: Tensor2D) => [Tensor2D[], Tensor1D];
  inverse: (z: Tensor2D) => [Tensor2D[], Tensor1D];
} {
  const modelUnknown = model as unknown as Record<string, unknown>;
  return 'forward' in model && 'inverse' in model &&
         typeof modelUnknown.forward === 'function' &&
         typeof modelUnknown.inverse === 'function';
}

/**
 * Train the flow-based model
 * Returns the trained model
 */
export async function trainModel<T extends FlowModel = FlowModel>(
  state: PipelineState,
  modelFactory: ModelFactory<T>,
  trainingWidget?: TrainingWidget
): Promise<T> {
  console.log('Starting training...');

  // Try WebGPU first (fastest), fall back to WebGL
  try {
    await tf.setBackend('webgpu');
    await tf.ready();
    console.log('Using WebGPU backend (fastest)');
  } catch {
    console.log('WebGPU not available, falling back to WebGL');
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('Using WebGL backend');

    // Enable WebGL performance optimizations
    tf.env().set('WEBGL_PACK', true);
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
  }

  console.log('Active backend:', tf.getBackend());

  // Use existing model from state (for resume) or create new one
  const flow = (state.model ?? modelFactory()) as T;
  if (state.model === null) {
    console.log('Created new model using factory function');
  } else {
    console.log('Resuming training with existing model');
  }

  // Training parameters
  const numEpochs = state.numEpochs;
  const batchSize = 256;
  const maxLr = 0.001;
  const minLr = 0.0001;
  const warmupEpochs = 100; // Linear warmup for first 100 epochs
  const clipNorm = 1.0; // Gradient clipping threshold

  // Create optimizer (will be recreated with learning rate schedule)
  let optimizer = tf.train.adam(minLr);

  // Set max epochs for training widget
  if (trainingWidget) {
    trainingWidget.setMaxEpochs(numEpochs);
  }

  // Track timing for ETA
  const startTime = Date.now();
  let lastLogTime = startTime;
  const timings: number[] = [];

  // Get existing loss history (for resume) or start fresh
  const lossHistory: Pair<number>[] = trainingWidget ? trainingWidget.getLossHistory() : [];
  const startEpoch = lossHistory.length;

  console.log(`Starting from epoch ${startEpoch}`);

  // Training loop
  for (let epoch = startEpoch; epoch < numEpochs; epoch++) {
    // Check if training should be paused
    if (state.trainingState !== 'training') {
      console.log('Training paused at epoch', epoch);
      state.trainingState = 'paused';
      return flow;
    }

    // Learning rate schedule with warmup + cosine annealing
    let currentLr: number;
    if (epoch < warmupEpochs) {
      // Linear warmup: 0.0001 -> 0.01 over first 100 epochs
      currentLr = minLr + (maxLr - minLr) * (epoch / warmupEpochs);
    } else {
      // Cosine annealing after warmup
      const progress = (epoch - warmupEpochs) / (numEpochs - warmupEpochs);
      currentLr = minLr + 0.5 * (maxLr - minLr) * (1 + Math.cos(Math.PI * progress));
    }

    // Update optimizer with new learning rate every 10 epochs (for efficiency)
    if (epoch % 10 === 0 && epoch > 0) {
      optimizer.dispose();
      optimizer = tf.train.adam(currentLr);
    }

    // Generate batch of moons data
    const x = makeMoons(batchSize, 0.05);

    // Compute loss and gradients with clipping
    const { lossValue, clippedGrads } = tf.tidy(() => {
      // Compute gradients
      const { value: loss, grads } = tf.variableGrads(() => flow.computeLoss(x));

      // Compute global norm of gradients
      const gradValues = Object.values(grads);
      let sumSquares = tf.scalar(0);
      for (const grad of gradValues) {
        sumSquares = tf.add(sumSquares, tf.sum(tf.square(grad)));
      }
      const globalNorm = tf.sqrt(sumSquares);

      // Compute clipping coefficient
      const clipCoeff = tf.minimum(
        tf.scalar(1.0),
        tf.div(clipNorm, tf.add(globalNorm, 1e-6))
      );

      // Apply clipping to all gradients
      const clippedGrads: Record<string, tfjs.Tensor> = {};
      Object.keys(grads).forEach(name => {
        clippedGrads[name] = tf.mul(grads[name], clipCoeff);
      });

      return {
        lossValue: loss.dataSync() as Float32Array,
        clippedGrads
      };
    });

    // Apply clipped gradients
    // @ts-expect-error - TensorFlow.js type mismatch between gradient types
    optimizer.applyGradients(clippedGrads);

    // Add to loss history every epoch
    lossHistory.push([epoch, lossValue[0]]);

    // Update visualization every epoch
    if (trainingWidget) {
      trainingWidget.update(lossHistory);
    }

    // Log progress every 10 epochs
    if (epoch % 10 === 0) {
      const now = Date.now();
      const epochTime = (now - lastLogTime) / 10; // ms per epoch
      timings.push(epochTime);

      // Calculate ETA
      const remainingEpochs = numEpochs - epoch;
      const avgEpochTime = timings.reduce((a, b) => a + b, 0) / timings.length;
      const etaSeconds = (remainingEpochs * avgEpochTime) / 1000;
      const etaMinutes = Math.floor(etaSeconds / 60);
      const etaSecondsRemainder = Math.floor(etaSeconds % 60);

      const progress = ((epoch / numEpochs) * 100).toFixed(1);
      console.log(
        `Epoch ${epoch}/${numEpochs} (${progress}%) - ` +
        `Loss: ${lossValue[0].toFixed(4)} - ` +
        `LR: ${currentLr.toFixed(6)} - ` +
        `${epochTime.toFixed(1)}ms/epoch - ` +
        `ETA: ${etaMinutes}m ${etaSecondsRemainder}s`
      );

      lastLogTime = now;

      // Keep only last 5 timings for moving average
      if (timings.length > 5) {
        timings.shift();
      }
    }

    // Cleanup
    x.dispose();

    // Yield to browser every epoch for smooth visualization
    await tf.nextFrame();
  }

  console.log('Training complete!');
  state.trainingState = 'completed';

  // Test the trained model (only for models with forward/inverse)
  if (hasForwardInverse(flow)) {
    console.log('\nTesting trained model...');
    const testData = makeMoons(100, 0.05);
    const [zs] = flow.forward(testData);
    const z = zs[zs.length - 1];

    console.log('Original data (first 5 samples):');
    console.log(testData.slice([0, 0], [5, 2]).arraySync());
    console.log('\nTransformed data (first 5 samples):');
    console.log(z.slice([0, 0], [5, 2]).arraySync());

    // Test inverse
    const [xs] = flow.inverse(z);
    const reconstructed = xs[xs.length - 1];
    const reconstructionError = tf.mean(tf.abs(tf.sub(testData, reconstructed)));
    console.log('\nReconstruction error:', (await reconstructionError.data())[0]);

    // Cleanup
    testData.dispose();
    reconstructionError.dispose();
  }

  return flow;
}
