import type { Tensor2D } from './tf-types';

/**
 * Generate a 2D moons dataset (two interleaving half circles)
 * Similar to sklearn.datasets.make_moons
 */
export function makeMoons(nSamples: number, noise = 0.05): Tensor2D {
  const samplesPerMoon = Math.floor(nSamples / 2);

  // Generate outer moon (top half circle)
  const outerAngles = tf.linspace(0, Math.PI, samplesPerMoon);
  const outerX = tf.cos(outerAngles);
  const outerY = tf.sin(outerAngles);

  // Generate inner moon (bottom half circle, flipped and shifted)
  const innerAngles = tf.linspace(0, Math.PI, nSamples - samplesPerMoon);
  const innerX = tf.sub(1, tf.cos(innerAngles));
  const innerY = tf.sub(tf.mul(tf.sin(innerAngles), -1), -0.5);

  // Combine moons
  const x = tf.concat([outerX, innerX]);
  const y = tf.concat([outerY, innerY]);

  // Stack into [n, 2] tensor
  let moons = tf.stack([x, y], 1);

  // Add noise if specified
  if (noise > 0) {
    const noiseVals = tf.randomNormal([nSamples, 2], 0, noise);
    moons = tf.add(moons, noiseVals);
    noiseVals.dispose();
  }

  // Cleanup intermediate tensors
  outerAngles.dispose();
  outerX.dispose();
  outerY.dispose();
  innerAngles.dispose();
  innerX.dispose();
  innerY.dispose();
  x.dispose();
  y.dispose();

  return moons as Tensor2D;
}
