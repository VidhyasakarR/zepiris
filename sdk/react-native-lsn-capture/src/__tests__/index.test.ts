const mockNative = {
  warmUp: jest.fn(),
  capture: jest.fn(),
  score: jest.fn(),
};
const mockRN: { Platform: { OS: string }; NativeModules: Record<string, unknown> } = {
  Platform: { OS: 'android' },
  NativeModules: { LsnCapture: mockNative },
};
jest.mock('react-native', () => mockRN);

type Sdk = typeof import('../index');
const load = (): Sdk => {
  let sdk: Sdk | undefined;
  jest.isolateModules(() => {
    sdk = require('../index');
  });
  return sdk!;
};

const scoreOpts = {
  apiBase: 'https://x.io',
  sourceSelfieS3: 'https://b/r.jpg',
};
const nativeErr = (code: string, message = code) => Object.assign(new Error(message), { code });

beforeEach(() => {
  jest.resetAllMocks();
  mockRN.Platform.OS = 'android';
  mockRN.NativeModules = { LsnCapture: mockNative };
});

describe('platform support', () => {
  it('is supported on Android with the module linked', () => {
    expect(load().isSupported()).toBe(true);
  });

  it.each([
    ['iOS', () => { mockRN.Platform.OS = 'ios'; }],
    ['no module (Expo Go / not linked)', () => { mockRN.NativeModules = {}; }],
    ['a module without methods', () => { mockRN.NativeModules = { LsnCapture: {} }; }],
    ['NativeModules missing', () => { (mockRN as { NativeModules?: unknown }).NativeModules = undefined; }],
  ])('is unsupported with %s', async (_label, setup) => {
    setup();
    const sdk = load();
    expect(sdk.isSupported()).toBe(false);
    await expect(sdk.capture()).rejects.toMatchObject({ code: 'unsupported_platform' });
    await expect(sdk.warmUp()).rejects.toMatchObject({ code: 'unsupported_platform' });
    await expect(sdk.score({ ...scoreOpts, faceCheckPath: '/x.jpg' })).rejects.toMatchObject({
      code: 'unsupported_platform',
    });
    await expect(sdk.start(scoreOpts)).rejects.toMatchObject({ code: 'unsupported_platform' });
  });
});

describe('capture', () => {
  const shot = { path: '/c/lsn_capture_1.jpg', uri: 'file:///c/lsn_capture_1.jpg', challenge: 'blink', challengePassed: true, stats: { frames: 3 } };

  it('fills in defaults', async () => {
    mockNative.capture.mockResolvedValue(shot);
    await expect(load().capture()).resolves.toEqual(shot);
    expect(mockNative.capture).toHaveBeenCalledWith({
      challenge: 'blink', light: false, maxSide: 2592, jpegQuality: 92, brightness: 0,
    });
  });

  it('coerces bad options instead of throwing', async () => {
    mockNative.capture.mockResolvedValue(shot);
    const sdk = load();
    await sdk.capture({ challenge: ' TURN ' as never, maxSide: NaN, jpegQuality: 150, brightness: -9, light: 'yes' as never });
    expect(mockNative.capture).toHaveBeenLastCalledWith({
      challenge: 'turn', light: false, maxSide: 2592, jpegQuality: 100, brightness: -2,
    });
    await sdk.capture({ challenge: 'wink' as never, maxSide: -5, jpegQuality: 10 });
    expect(mockNative.capture).toHaveBeenLastCalledWith({
      challenge: 'blink', light: false, maxSide: 0, jpegQuality: 60, brightness: 0,
    });
    await sdk.capture(null as never);
    expect(mockNative.capture).toHaveBeenLastCalledWith(expect.objectContaining({ challenge: 'blink' }));
  });

  it('fills a missing uri / stats', async () => {
    mockNative.capture.mockResolvedValue({ path: '/c/a.jpg', challenge: 'none', challengePassed: 0 });
    await expect(load().capture({ challenge: 'none' })).resolves.toEqual({
      path: '/c/a.jpg', uri: 'file:///c/a.jpg', challenge: 'none', challengePassed: false, stats: {},
    });
  });

  it('passes native codes through and maps unknown ones to capture_failed', async () => {
    const sdk = load();
    mockNative.capture.mockRejectedValueOnce(nativeErr('cancelled'));
    await expect(sdk.capture()).rejects.toMatchObject({ code: 'cancelled' });
    mockNative.capture.mockRejectedValueOnce(nativeErr('E_WEIRD', 'native detail'));
    await expect(sdk.capture()).rejects.toMatchObject({ code: 'capture_failed', nativeCode: 'E_WEIRD', message: 'native detail' });
  });
});

describe('score', () => {
  it('validates in JS before calling native', async () => {
    const sdk = load();
    await expect(sdk.score({ ...scoreOpts })).rejects.toMatchObject({ code: 'invalid_config' });
    await expect(sdk.score({ ...scoreOpts, faceCheckPath: '/x.jpg', apiBase: 'nope' })).rejects.toMatchObject({ code: 'invalid_config' });
    await expect(sdk.score(undefined as never)).rejects.toMatchObject({ code: 'invalid_config' });
    expect(mockNative.score).not.toHaveBeenCalled();
  });

  it('sends defaults and maps the reply', async () => {
    mockNative.score.mockResolvedValue({ requestId: 'r1', scores: { logo: { score: 0.9 } } });
    const s = await load().score({ ...scoreOpts, faceCheckPath: '/x.jpg' });
    expect(s.requestId).toBe('r1');
    expect(s.logo).toBe(0.9);
    expect(mockNative.score).toHaveBeenCalledWith({
      apiBase: 'https://x.io',
      checks: ['face_match', 'dress_color', 'logo'],
      faceCheckPath: '/x.jpg',
      faceCheckS3: undefined,
      sourceSelfieS3: 'https://b/r.jpg',
      sourceSelfieB64: undefined,
      timeoutMs: 40000,
      headers: {},
    });
  });

  it('maps unknown native failures to network', async () => {
    mockNative.score.mockRejectedValueOnce(new Error('socket closed'));
    await expect(load().score({ ...scoreOpts, faceCheckPath: '/x.jpg' })).rejects.toMatchObject({
      code: 'network', message: 'socket closed',
    });
  });
});

describe('start', () => {
  it('validates the score options before opening the camera', async () => {
    await expect(load().start({ apiBase: 'https://x.io' })).rejects.toMatchObject({ code: 'invalid_config' });
    expect(mockNative.capture).not.toHaveBeenCalled();
  });

  it('scores the captured photo, ignoring any photo source passed in', async () => {
    mockNative.capture.mockResolvedValue({ path: '/c/shot.jpg', uri: 'file:///c/shot.jpg', challenge: 'turn', challengePassed: true, stats: {} });
    mockNative.score.mockResolvedValue({ requestId: 'r2', scores: {} });
    const opts = { ...scoreOpts, challenge: 'random' as const, faceCheckS3: 'https://b/other.jpg' };
    const r = await load().start(opts as never);
    expect(r.capture.path).toBe('/c/shot.jpg');
    expect(r.scores.requestId).toBe('r2');
    expect(mockNative.capture).toHaveBeenCalledWith(expect.objectContaining({ challenge: 'random' }));
    const sent = mockNative.score.mock.calls[0][0];
    expect(sent.faceCheckPath).toBe('/c/shot.jpg');
    expect(sent.faceCheckS3).toBeUndefined();
    expect(sent).not.toHaveProperty('challenge');
  });

  it('does not score when the rider backs out', async () => {
    mockNative.capture.mockRejectedValue(nativeErr('cancelled'));
    await expect(load().start(scoreOpts)).rejects.toMatchObject({ code: 'cancelled' });
    expect(mockNative.score).not.toHaveBeenCalled();
  });
});
