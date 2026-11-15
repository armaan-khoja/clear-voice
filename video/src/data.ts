import raw from './data.json';

export type Token = {w: string; tag: 'ok' | 'sub' | 'ins'};
export type ClipSystem = {text: string; tokens: Token[]; wer: number};
export type Clip = {
	rowId: number;
	file: string; // under public/, e.g. clips/123.wav
	spec?: string; // spectrogram image under public/, e.g. spec/123.png
	speaker: string;
	severity: 'mild' | 'moderate' | 'moderate-severe' | 'severe';
	duration: number; // seconds
	ref: string;
	systems: Record<string, ClipSystem>; // stockName, 'whisper-large-v3', 'Clear Voice'
};
export type Bar = {name: string; value: number; lo?: number; hi?: number};
export type PersonalSeries = {speaker: string; severity: string; points: {minutes: number; wer: number}[]};

export type VideoData = {
	placeholder: boolean; // true until scripts/video_data.py writes real results
	stockName: string; // the stock model Clear Voice was fine-tuned from, e.g. 'whisper-medium'
	largeRatio: number; // whisper-large-v3's size relative to stockName, rounded
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
	stockTypical?: number; // stock model's WER on typical (control) speakers, same sentences
	librispeech: Bar[];
	personalisation: PersonalSeries[];
	opening: Clip;
	demo: Clip[];
	hasVoiceover: boolean;
};

export const data = raw as unknown as VideoData;

/** The stock model Clear Voice started from, and how much bigger whisper-large-v3 is. */
export const STOCK = data.stockName ?? 'whisper-small';
export const BIGGER = `${data.largeRatio ?? 6}×`;
export const BIGGER_WORDS = ({2: 'twice', 3: 'three times', 6: 'six times'} as Record<number, string>)[data.largeRatio ?? 6] ?? `${data.largeRatio} times`;

/** Single-word character error rate per system, by name. */
export const wordCer = (name: string) => data.headlineWords.find((b) => b.name === name)?.value;

/** Severity as words, e.g. 'moderate-severe' -> 'moderate to severe'. */
export const severityWords = (s: string) => (s === 'moderate-severe' ? 'moderate to severe' : s);

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
