import {AbsoluteFill, Audio, interpolate, Sequence, staticFile} from 'remotion';
import {data} from '../data';
import type {Part, SceneKind} from './timeline';
import {parts, speechWindows, starts, totalFrames} from './timeline';

export type SceneProps = {part: Part};

type Sfx = {file: string; volume: number; offset?: number};

// Lays the scenes on the shared timeline and mixes the sound: music bed ducked under every real
// recording, the recordings themselves, an optional voiceover, and the look's own page-turn sound.
export const Film: React.FC<{
	scenes: Record<SceneKind, React.FC<SceneProps>>;
	ground: string;
	music?: {file: string; volume: number};
	turn?: Sfx;
}> = ({scenes, ground, music, turn}) => {
	const ps = parts();
	const s = starts(ps);
	const total = totalFrames();
	const windows = speechWindows();
	const pro = ps.findIndex((p) => p.kind === 'prologue');
	const musicFrom = pro < 0 ? 0 : s[pro] + ps[pro].frames - 20;
	const musicVolume = (f: number) => {
		const fade = interpolate(f, [0, 45, total - 75, total], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
		const duck = windows.reduce(
			(v, [a, b]) => Math.min(v, interpolate(f, [a - 12, a, b, b + 15], [1, 0.12, 0.12, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})),
			1,
		);
		// No music under the prologue and epilogue: the recording and the silence carry those scenes.
		const quiet = ps.reduce((v, p, i) => {
			if (p.kind !== 'prologue' && p.kind !== 'epilogue') return v;
			const [a, b] = [s[i], s[i] + p.frames];
			return Math.min(v, interpolate(f, [a - 30, a, b - 20, b + 40], [1, 0, 0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
		}, 1);
		return (music?.volume ?? 0) * (data.hasVoiceover ? 0.6 : 1) * fade * duck * quiet;
	};
	return (
		<AbsoluteFill style={{backgroundColor: ground}}>
			{music ? (
				// The bed starts as the exam begins, so its own fade-out lands where the quiet epilogue starts.
				<Sequence from={musicFrom} layout="none">
					<Audio src={staticFile(music.file)} volume={(f) => musicVolume(f + musicFrom)} />
				</Sequence>
			) : null}
			{data.hasVoiceover ? <Audio src={staticFile('voiceover.wav')} /> : null}
			{ps.map((p, i) => {
				const Scene = scenes[p.kind];
				return (
					<Sequence key={p.key} from={s[i]} durationInFrames={p.frames} name={p.key}>
						<Scene part={p} />
						{p.clip && p.playAt !== undefined ? (
							<Sequence from={p.playAt} layout="none">
								<Audio src={staticFile(p.clip.file)} />
							</Sequence>
						) : null}
					</Sequence>
				);
			})}
			{turn
				? s.slice(1).map((st, i) => ps[i + 1].kind === 'epilogue' ? null : (
						<Sequence key={`turn${i}`} from={st + (turn.offset ?? 0)} durationInFrames={40} layout="none">
							<Audio src={staticFile(turn.file)} volume={turn.volume} />
						</Sequence>
					))
				: null}
		</AbsoluteFill>
	);
};

export const Watermark: React.FC<{color: string; font: string}> = ({color, font}) =>
	data.placeholder ? (
		<div style={{position: 'absolute', top: 18, left: 0, right: 0, textAlign: 'center', fontFamily: font, fontSize: 20, fontWeight: 700, color, letterSpacing: '0.04em'}}>
			Preview with placeholder data
		</div>
	) : null;
