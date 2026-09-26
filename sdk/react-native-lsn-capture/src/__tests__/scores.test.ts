import { toScores } from '../scores';

const full = {
  requestId: 'req-123',
  scoredAt: '2026-09-26T10:00:00Z',
  checksRequested: ['face_match', 'dress_color', 'logo'],
  scores: {
    face_match: { similarity: 0.83, liveness: 0.97, faceDetected: true },
    dress_color: { score: 0.61 },
    logo: {
      score: 0.9,
      read: { text: 'AMAZON', matched: true, onBlue: true, reason: null },
    },
  },
  image: { sha256: 'abc123', bytes: 812345 },
};

describe('toScores', () => {
  it('flattens a full response', () => {
    const s = toScores(full);
    expect(s).toEqual({
      requestId: 'req-123',
      scoredAt: '2026-09-26T10:00:00Z',
      checksRequested: ['face_match', 'dress_color', 'logo'],
      faceSimilarity: 0.83,
      liveness: 0.97,
      faceDetected: true,
      dressColor: 0.61,
      logo: 0.9,
      logoRead: { text: 'AMAZON', matched: true, onBlue: true, reason: null },
      imageSha256: 'abc123',
      raw: full,
    });
    expect(s.raw).toBe(full);
  });

  it('returns nulls for checks that were not requested', () => {
    const raw = {
      requestId: 'req-2',
      checksRequested: ['face_match'],
      scores: { face_match: { similarity: 0.7, liveness: null, faceDetected: true } },
    };
    const s = toScores(raw);
    expect(s.checksRequested).toEqual(['face_match']);
    expect(s.faceSimilarity).toBe(0.7);
    expect(s.liveness).toBeNull();
    expect(s.dressColor).toBeNull();
    expect(s.logo).toBeNull();
    expect(s.logoRead).toBeNull();
    expect(s.imageSha256).toBeNull();
    expect(s.scoredAt).toBeNull();
  });

  it('keeps a null similarity null when no face was found', () => {
    const s = toScores({
      requestId: 'req-3',
      checksRequested: ['face_match', 'dress_color'],
      scores: {
        face_match: { similarity: null, liveness: null, faceDetected: false },
        dress_color: { score: 0 },
      },
    });
    expect(s.faceSimilarity).toBeNull();
    expect(s.faceDetected).toBe(false);
    expect(s.dressColor).toBe(0); // 0 is a real score, not "missing"
  });

  it('exposes what the logo check read', () => {
    const s = toScores({
      requestId: 'req-4',
      checksRequested: ['logo'],
      scores: {
        logo: { score: 0.2, read: { text: 'AMAZ0N', matched: false, onBlue: false, reason: 'no_blue' } },
      },
    });
    expect(s.logo).toBe(0.2);
    expect(s.logoRead).toEqual({ text: 'AMAZ0N', matched: false, onBlue: false, reason: 'no_blue' });
    expect(s.faceSimilarity).toBeNull();
    expect(s.faceDetected).toBeNull();
  });

  it('never throws on an empty or odd response', () => {
    const s = toScores({ scores: 'nope', checksRequested: 'x', image: [] } as Record<string, unknown>);
    expect(s.requestId).toBe('');
    expect(s.checksRequested).toEqual([]);
    expect(s.faceSimilarity).toBeNull();
    expect(s.imageSha256).toBeNull();
  });
});
