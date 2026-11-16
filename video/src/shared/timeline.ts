import type {Clip} from '../data';
import {data} from '../data';
import T from './timeline.json';

// One timeline for all three looks, so the team's voiceover (SCRIPT.md) fits whichever is chosen.
// Scene lengths live in timeline.json so the narration script (scripts/video_script.py) uses the same timings.
export const FPS = T.fps;
export const OVERLAP = T.overlap; // frames each scene overlaps the one before it (its entrance plays over the previous scene)
export const HOOK_PLAY = T.hookPlay; // frame within the hook where the opening recording starts
export const PROLOGUE_PLAY = T.prologuePlay; // frame within the prologue where its recording starts
export const DEMO_PLAY = T.demoPlay; // frame within each demo scene where its recording starts
const F = T.frames;
const AFTER_CLIP = T.afterClip; // frames a demo scene stays on after its recording ends

// A demo scene lasts its usual length, or longer when the recording is long (slow, severe speech).
export const demoFrames = (clip: Clip): number => Math.max(F.demo, DEMO_PLAY + Math.round(clip.duration * FPS) + AFTER_CLIP);

export type SceneKind = 'prologue' | 'epilogue' | 'figures' | 'title' | 'hook' | 'problem' | 'demo' | 'how' | 'results' | 'honest' | 'personal' | 'close';

export type Part = {
	key: string;
	kind: SceneKind;
	frames: number;
	clip?: Clip;
	index?: number;
	total?: number;
	playAt?: number; // frame within the scene where its recording starts
};

export const demoClips = (): Clip[] => (data.demo.length ? data.demo : [data.opening]);

export const parts = (): Part[] => {
	const demos = demoClips();
	const total = demos.length;
	const list: Part[] = [
		{key: 'prologue', kind: 'prologue', frames: F.prologue, clip: data.opening, playAt: PROLOGUE_PLAY},
		{key: 'title', kind: 'title', frames: F.title},
		{key: 'hook', kind: 'hook', frames: F.hook, clip: data.opening, playAt: HOOK_PLAY},
		{key: 'problem', kind: 'problem', frames: F.problem},
		...demos.map((clip, i) => ({key: `demo${i}`, kind: 'demo' as const, frames: demoFrames(clip), clip, index: i, total: demos.length, playAt: DEMO_PLAY})),
		{key: 'how', kind: 'how', frames: F.how},
		{key: 'results', kind: 'results', frames: F.results},
		{key: 'honest', kind: 'honest', frames: F.honest},
		{key: 'personal', kind: 'personal', frames: F.personal},
		{key: 'figures', kind: 'figures', frames: F.figures},
		{key: 'epilogue', kind: 'epilogue', frames: F.epilogue},
		{key: 'close', kind: 'close', frames: F.close},
	];
	return list.map((p) => ({...p, total}));
};

export const starts = (ps: Part[] = parts()): number[] =>
	ps.reduce<number[]>((acc, p, i) => [...acc, i ? acc[i - 1] + ps[i - 1].frames - OVERLAP : 0], []);

export const totalFrames = (): number => {
	const ps = parts();
	const s = starts(ps);
	return s[s.length - 1] + ps[ps.length - 1].frames;
};

// Absolute [start, end] frames of every real recording, for ducking music under the speaker.
export const speechWindows = (): [number, number][] => {
	const ps = parts();
	const s = starts(ps);
	return ps
		.map((p, i) => (p.clip && p.playAt !== undefined ? ([s[i] + p.playAt, s[i] + p.playAt + Math.round(p.clip.duration * FPS)] as [number, number]) : null))
		.filter((w): w is [number, number] => w !== null);
};

export const clipEnd = (p: {clip?: Clip; playAt?: number}): number => (p.playAt ?? 0) + Math.round((p.clip?.duration ?? 0) * FPS);
