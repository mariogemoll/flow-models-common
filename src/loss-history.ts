import { dequantizeFloats, quantizeFloats } from 'web-ui-common/data';
import type { Pair } from 'web-ui-common/types';

/**
 * Save loss history to a compressed binary file for download
 * Format: 8 byte header (min/max) + quantized uint8 values
 */
export function saveLossHistory(lossHistory: Pair<number>[]): void {
  // Extract just the loss values (epoch numbers are implicit: 0, 1, 2, ...)
  const lossValues = lossHistory.map(([, loss]) => loss);
  const float32Array = new Float32Array(lossValues);

  // Quantize to uint8
  const compressed = quantizeFloats(float32Array);

  const blob = new Blob([compressed], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = 'loss-history.bin';
  a.click();

  URL.revokeObjectURL(url);
  console.log('Loss history saved! Check your downloads folder for loss-history.bin');
}

/**
 * Load loss history from a compressed binary file
 * Format: 8 byte header (min/max) + quantized uint8 values
 */
export async function loadLossHistory(path: string): Promise<Pair<number>[] | null> {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();

    // Expand from binary format
    const result = dequantizeFloats(buffer);
    const lossValues = result[2];

    // Convert to [epoch, loss] pairs
    const data: Pair<number>[] = Array.from(lossValues).map((loss, epoch) => [epoch, loss]);

    console.log(`Loaded loss history with ${data.length} epochs`);
    return data;
  } catch (error) {
    console.error('Failed to load loss history:', error);
    return null;
  }
}
