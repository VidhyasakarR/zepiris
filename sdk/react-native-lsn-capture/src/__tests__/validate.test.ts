import { MAX_SELFIE_B64_CHARS, validateScoreOptions } from '../scores';
import type { LsnScoreOptions } from '../types';

const base: LsnScoreOptions = {
  apiBase: 'https://3-108-193-187.sslip.io/',
  sourceSelfieS3: 'https://b.s3.amazonaws.com/r.jpg',
};

const invalid = (opts: unknown, requirePhoto = false) => {
  try {
    validateScoreOptions(opts as LsnScoreOptions, requirePhoto);
  } catch (e) {
    expect((e as { code?: string }).code).toBe('invalid_config');
    return (e as Error).message;
  }
  throw new Error('expected invalid_config');
};

describe('validateScoreOptions', () => {
  it('accepts the defaults', () => {
    expect(() => validateScoreOptions(base)).not.toThrow();
    expect(() => validateScoreOptions({ ...base, apiBase: 'http://10.0.0.1:8000/api/' })).not.toThrow();
  });

  it('rejects missing options', () => {
    invalid(undefined);
    invalid(null);
  });

  it.each([
    [''],
    ['ftp://x.io'],
    ['https://'],
    ['x.io'],
    ['https://x.io/?token=1'],
    ['https://x.io/#frag'],
    ['https://x io'],
  ])('rejects apiBase %p', (apiBase) => {
    invalid({ ...base, apiBase });
  });

  it('rejects a non-string apiBase', () => {
    invalid({ ...base, apiBase: 42 });
  });

  it('validates checks', () => {
    invalid({ ...base, checks: [] });
    invalid({ ...base, checks: ['face_match', 'hat'] });
    invalid({ ...base, checks: 'logo' });
    expect(() => validateScoreOptions({ apiBase: base.apiBase, checks: ['logo'] })).not.toThrow();
  });

  it('needs a selfie for face_match', () => {
    invalid({ apiBase: base.apiBase });
    invalid({ apiBase: base.apiBase, sourceSelfieS3: '' });
    expect(() => validateScoreOptions({ apiBase: base.apiBase, sourceSelfieB64: 'abc' })).not.toThrow();
  });

  it('caps sourceSelfieB64', () => {
    invalid({ apiBase: base.apiBase, sourceSelfieB64: 'A'.repeat(MAX_SELFIE_B64_CHARS + 1) });
  });

  it('checks the photo source only when asked', () => {
    expect(() => validateScoreOptions(base)).not.toThrow();
    invalid(base, true);
    invalid({ ...base, faceCheckPath: '/x.jpg', faceCheckS3: 'https://b/x.jpg' }, true);
    expect(() => validateScoreOptions({ ...base, faceCheckPath: '/x.jpg' }, true)).not.toThrow();
    expect(() => validateScoreOptions({ ...base, faceCheckS3: 'https://b/x.jpg' }, true)).not.toThrow();
  });

  it.each([[0], [-1], [NaN], [Infinity], [600001], ['40000']])('rejects timeoutMs %p', (timeoutMs) => {
    invalid({ ...base, timeoutMs });
  });

  it('accepts timeoutMs in range', () => {
    expect(() => validateScoreOptions({ ...base, timeoutMs: 1 })).not.toThrow();
    expect(() => validateScoreOptions({ ...base, timeoutMs: 600000 })).not.toThrow();
  });

  it('validates headers without echoing values', () => {
    const msg = invalid({ ...base, headers: { 'X-Token': 'secret\r\nX-Evil: 1' } });
    expect(msg).not.toContain('secret');
    invalid({ ...base, headers: { 'X Token': 'v' } });
    invalid({ ...base, headers: { '': 'v' } });
    invalid({ ...base, headers: { 'X-Token': 'tökén' } });
    invalid({ ...base, headers: { 'X-Token': 5 } });
    invalid({ ...base, headers: 'Authorization: x' });
    expect(() =>
      validateScoreOptions({ ...base, headers: { Authorization: 'Bearer a.b-c', 'X-T': 'a\tb' } }),
    ).not.toThrow();
  });
});
