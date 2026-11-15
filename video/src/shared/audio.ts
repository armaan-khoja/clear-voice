import {getWaveformPortion, useAudioData} from '@remotion/media-utils';
import {interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';

// The clip's real amplitude envelope (0..1, normalised to its peak) and the playhead's progress.
export const useWave = (file: string, duration: number, playFrom: number, samples: number) => {
	const f = useCurrentFrame();
	const {fps} = useVideoConfig();
	const audio = useAudioData(staticFile(file));
	const progress = interpolate(f, [playFrom, playFrom + duration * fps], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
	if (!audio) return {bars: [] as number[], progress};
	const portion = getWaveformPortion({audioData: audio, startTimeInSeconds: 0, durationInSeconds: Math.min(duration, audio.durationInSeconds), numberOfSamples: samples});
	const peak = Math.max(...portion.map((b) => b.amplitude), 1e-6);
	return {bars: portion.map((b) => Math.max(0.03, Math.pow(b.amplitude / peak, 0.75))), progress};
};

export const easeOut = (t: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 4); // quart: settles without overshoot

export const pct = (x: number) => `${Math.round(x * 100)}%`;
