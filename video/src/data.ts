import raw from './data.json';

export type Token = {w: string; tag: 'ok' | 'sub' | 'ins'};
export type ClipSystem = {text: string; tokens: Token[]; wer: number};
export type Clip = {
	rowId: number;
	file: string; // under public/, e.g. clips/123.wav
	spec?: string; // spectrogram image under public/, e.g. spec/123.png
	speaker: string;
	severity: 'mild' | 'moderate' | 'severe';
	duration: number; // seconds
	ref: string;
	systems: Record<string, ClipSystem>; // 'whisper-small', 'whisper-large-v3', 'Clear Voice'
};
export type Bar = {name: string; value: number; lo?: number; hi?: number};
export type PersonalSeries = {speaker: string; severity: string; points: {minutes: number; wer: number}[]};

export type VideoData = {
	placeholder: boolean; // true until scripts/video_data.py writes real results
	team: string;
	repoUrl: string;
	demoUrl: string;
	nRecordings: number;
	nSpeakers: number;
	nDysarthric: number;
	trainMinutes: number; // GPU minutes for one fold
	headline: Bar[]; // sentence WER, dysarthric speakers, new speaker + new sentence
	headlineWords: Bar[]; // single-word CER, same speakers
	leakage: Bar[]; // Clear Voice WER under each test condition
	stockTypical?: number; // stock whisper-small WER on typical (control) speakers, same sentences
	librispeech: Bar[];
	personalisation: PersonalSeries[];
	opening: Clip;
	demo: Clip[];
	hasVoiceover: boolean;
};

export const data = raw as unknown as VideoData;

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
