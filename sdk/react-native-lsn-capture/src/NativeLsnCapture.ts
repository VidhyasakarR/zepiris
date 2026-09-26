import { NativeModules, Platform } from 'react-native';
import type { LsnCaptureResult, LsnWarmUpResult } from './types';

/**
 * The `LsnCapture` legacy native module (LsnCaptureModule.kt). It works under
 * the new architecture through the interop layer. Android only.
 */
export interface NativeLsnCaptureSpec {
  warmUp(): Promise<LsnWarmUpResult>;
  capture(opts: Record<string, unknown>): Promise<LsnCaptureResult>;
  score(opts: Record<string, unknown>): Promise<Record<string, unknown>>;
}

function lookup(): NativeLsnCaptureSpec | null {
  try {
    if (Platform?.OS !== 'android') return null;
    const mod: NativeLsnCaptureSpec | undefined = NativeModules?.LsnCapture;
    return mod && typeof mod.capture === 'function' ? mod : null;
  } catch {
    return null; // e.g. a test / Expo Go environment without the native runtime
  }
}

/** null on iOS / web, or when the native module is not linked. */
export const NativeLsnCapture: NativeLsnCaptureSpec | null = lookup();
