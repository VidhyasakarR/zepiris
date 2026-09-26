import { LSN_ERROR_CODES, lsnError, toLsnError } from '../errors';

describe('lsnError', () => {
  it('builds an Error with name, code and message', () => {
    const e = lsnError('busy', 'a capture is already running');
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe('LsnError');
    expect(e.code).toBe('busy');
    expect(e.message).toBe('a capture is already running');
  });

  it('uses the code as the message when none is given', () => {
    expect(lsnError('timeout').message).toBe('timeout');
  });
});

describe('toLsnError', () => {
  it('keeps every known native code', () => {
    for (const code of LSN_ERROR_CODES) {
      const native = Object.assign(new Error('boom'), { code });
      const e = toLsnError(native, 'network');
      expect(e.code).toBe(code);
      expect(e.nativeCode).toBeUndefined();
      expect(e).toBe(native); // same object: native stack / userInfo preserved
    }
  });

  it('maps an unknown native code to the fallback, keeping message and original code', () => {
    const native = Object.assign(new Error('Something native'), { code: 'E_UNKNOWN' });
    const e = toLsnError(native, 'capture_failed');
    expect(e.code).toBe('capture_failed');
    expect(e.nativeCode).toBe('E_UNKNOWN');
    expect(e.message).toBe('Something native');
  });

  it('handles an Error without a code', () => {
    const e = toLsnError(new TypeError('x is undefined'), 'network');
    expect(e.code).toBe('network');
    expect(e.nativeCode).toBeUndefined();
    expect(e.message).toBe('x is undefined');
  });

  it('handles plain objects, strings, null and undefined', () => {
    expect(toLsnError({ code: 'timeout', message: 'slow' }, 'network')).toMatchObject({
      code: 'timeout',
      message: 'slow',
    });
    expect(toLsnError('oops', 'server')).toMatchObject({ code: 'server', message: 'oops' });
    expect(toLsnError(null, 'server')).toMatchObject({ code: 'server', message: 'server' });
    expect(toLsnError(undefined, 'busy')).toMatchObject({ code: 'busy', message: 'busy' });
    expect(toLsnError({ code: 42 }, 'server')).toMatchObject({ code: 'server', nativeCode: '42' });
    expect(toLsnError(null, 'server')).toBeInstanceOf(Error);
  });
});
