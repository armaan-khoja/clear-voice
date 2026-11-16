import {loadFont as loadCourier} from '@remotion/google-fonts/CourierPrime';
import {loadFont as loadFranklin} from '@remotion/google-fonts/LibreFranklin';
import {loadFont as loadNewsreader} from '@remotion/google-fonts/Newsreader';
import {AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {data, STOCK} from '../data';
import {easeOut, useWave} from './audio';
import type {SceneProps} from './Film';
import {clipEnd} from './timeline';

// The film opens and closes on the person, not the model: a dark, quiet room, one real recording,
// and the words the software wrote. No music here; the voice and the silence after it carry the scene.

const {fontFamily: SERIF} = loadNewsreader('normal', {weights: ['400', '500'], subsets: ['latin']});
const {fontFamily: SANS} = loadFranklin('normal', {weights: ['400', '500'], subsets: ['latin']});
const {fontFamily: MONO} = loadCourier('normal', {weights: ['400'], subsets: ['latin']});

export const H = {
	night: '#121211',
	text: '#F1EEE6',
	dim: 'rgba(241,238,230,0.56)',
	faint: 'rgba(241,238,230,0.22)',
	red: '#E5534B',
	paper: '#fbfbf9',
};

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const COLUMN = 300; // left edge of the text column
const WIDTH = 1320;

// Fades and rises a few pixels, settling without overshoot; fades out again from `out` if given.
const Line: React.FC<{at: number; out?: number; style?: React.CSSProperties; children: React.ReactNode}> = ({at, out, style, children}) => {
	const f = useCurrentFrame();
	const t = easeOut((f - at) / 20);
	const o = out === undefined ? 1 : interpolate(f, [out, out + 18], [1, 0], clamp);
	return <div style={{position: 'absolute', left: COLUMN, width: WIDTH, opacity: t * o, transform: `translateY(${(1 - t) * 12}px)`, ...style}}>{children}</div>;
};

const Ground: React.FC<{fadeIn?: number; toPaper: number; children: React.ReactNode}> = ({fadeIn = 0, toPaper, children}) => {
	const f = useCurrentFrame();
	const dark = fadeIn ? interpolate(f, [0, fadeIn], [0, 1], clamp) : 1;
	const paper = interpolate(f, [toPaper, toPaper + 16], [0, 1], clamp);
	return (
		<AbsoluteFill style={{backgroundColor: H.paper}}>
			<AbsoluteFill style={{backgroundColor: H.night, opacity: dark}}>{children}</AbsoluteFill>
			<AbsoluteFill style={{backgroundColor: H.paper, opacity: paper}} />
		</AbsoluteFill>
	);
};

const he = (speaker: string) => (speaker.startsWith('F') ? {he: 'She', his: 'hers'} : {he: 'He', his: 'his'});

const NUMBER_WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve'];

// 100 words as 100 squares; the wrong ones fill in red, one after another.
const Hundred: React.FC<{label: string; wrong: number; at: number; note: string}> = ({label, wrong, at, note}) => {
	const f = useCurrentFrame();
	const filled = Math.max(0, Math.floor((f - at) / 2));
	const shown = Math.min(wrong, filled);
	return (
		<div style={{width: 420}}>
			<div style={{fontFamily: SANS, fontSize: 28, color: H.text, marginBottom: 22}}>{label}</div>
			<div style={{display: 'grid', gridTemplateColumns: 'repeat(10, 30px)', gap: 9}}>
				{Array.from({length: 100}, (_, i) => (
					<div key={i} style={{width: 30, height: 30, borderRadius: 2, border: `1.5px solid ${i < shown ? H.red : H.faint}`, background: i < shown ? H.red : 'transparent'}} />
				))}
			</div>
			<div style={{marginTop: 26, display: 'flex', alignItems: 'baseline', gap: 14}}>
				<span style={{fontFamily: SERIF, fontSize: 76, color: H.text, fontVariantNumeric: 'tabular-nums', minWidth: 80}}>{shown}</span>
				<span style={{fontFamily: SANS, fontSize: 24, color: H.dim}}>{note}</span>
			</div>
		</div>
	);
};

export const Prologue: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const clip = part.clip ?? data.opening;
	const playAt = part.playAt ?? 45;
	const end = clipEnd({clip, playAt});
	const heard = clip.systems[STOCK]?.text ?? '';
	const {he: He, his} = he(clip.speaker);

	const heardAt = end + 14;
	const typed = heard.slice(0, Math.max(0, Math.floor((f - heardAt - 8) / 3)));
	const saidAt = heardAt + 26 + heard.length * 3 + 36;
	const out1 = saidAt + 96;
	const statAt = out1 + 26;
	const gridAt = statAt + 34;
	const out2 = gridAt + 156;
	const lastAt = out2 + 24;

	const typical = Math.round(data.stockTypical * 100);
	const stock = data.headline.find((h) => h.name === STOCK);
	const dys = Math.round((stock?.value ?? 0) * 100);

	const {bars, progress} = useWave(clip.file, clip.duration, playAt, 110);
	const waveIn = interpolate(f, [16, 40], [0, 1], clamp);
	const waveOut = interpolate(f, [out1, out1 + 18], [1, 0], clamp);
	const caret = typed.length < heard.length || Math.floor(f / 15) % 2 === 0;

	return (
		<Ground toPaper={part.frames - 30}>
			<div style={{position: 'absolute', left: COLUMN, top: 250, width: WIDTH, height: 170, display: 'flex', alignItems: 'center', gap: 4, opacity: waveIn * waveOut}}>
				{bars.map((b, i) => (
					<div key={i} style={{flex: 1, height: `${b * 100}%`, borderRadius: 2, background: i / bars.length < progress ? H.text : H.faint}} />
				))}
			</div>
			<Line at={20} out={out1} style={{top: 448, fontFamily: SANS, fontSize: 22, color: H.dim}}>
				A recording from the TORGO speech database. Speaker {clip.speaker} has dysarthria.
			</Line>
			<Line at={heardAt} out={out1} style={{top: 560, display: 'flex', alignItems: 'baseline', gap: 28}}>
				<span style={{fontFamily: SANS, fontSize: 26, color: H.dim}}>Speech recognition wrote</span>
				<span style={{fontFamily: MONO, fontSize: 64, color: H.red}}>
					{typed}
					<span style={{opacity: caret ? 1 : 0, color: H.dim}}>|</span>
				</span>
			</Line>
			<Line at={saidAt} out={out1} style={{top: 690, fontFamily: SERIF, fontSize: 80, color: H.text, letterSpacing: '-0.01em'}}>
				{He} said “{clip.ref}.”
			</Line>

			<Line at={statAt} out={out2} style={{top: 150, fontFamily: SERIF, fontSize: 56, color: H.text, lineHeight: 1.2, width: 1500}}>
				Speech recognition is trained mostly on typical voices.
			</Line>
			<div style={{position: 'absolute', left: COLUMN, top: 300, display: 'flex', gap: 200, opacity: easeOut((f - gridAt) / 20) * interpolate(f, [out2, out2 + 18], [1, 0], clamp)}}>
				<Hundred label="Typical voices" wrong={typical} at={gridAt + 20} note="words wrong in every 100" />
				<Hundred label="Voices with dysarthria" wrong={dys} at={gridAt + 20} note="words wrong in every 100" />
			</div>
			<Line at={gridAt + 40} out={out2} style={{top: 930, fontFamily: SANS, fontSize: 20, color: H.dim}}>
				Stock {STOCK}, on TORGO speakers it was never trained on.
			</Line>

			<Line at={lastAt} style={{top: 400, fontFamily: SERIF, fontSize: 64, color: H.text, lineHeight: 1.25, width: 1250}}>
				So we trained it on voices like {his}, and set it an exam it had never seen.
			</Line>
		</Ground>
	);
};

export const Epilogue: React.FC<SceneProps> = ({part}) => {
	const n = NUMBER_WORDS[data.nDysarthric] ?? String(data.nDysarthric);
	return (
		<Ground fadeIn={14} toPaper={part.frames - 28}>
			<Line at={26} style={{top: 230, fontFamily: SERIF, fontSize: 60, color: H.text, lineHeight: 1.25}}>
				{n} people with dysarthria recorded these words for research.
			</Line>
			<Line at={112} style={{top: 425, fontFamily: SERIF, fontSize: 60, color: H.text, lineHeight: 1.25}}>
				Clear Voice got more of them right than the model it started from. Not all of them.
			</Line>
			<Line at={204} style={{top: 650, fontFamily: SERIF, fontSize: 60, color: H.dim, lineHeight: 1.25}}>
				The next test is with the people it is for, on their own words.
			</Line>
		</Ground>
	);
};

// The research paper's figures on a plain page, for the looks that have no exam insert of their own.
const PLAIN_FIGURES = ['paper/fig2-split.png', 'paper/fig4-headline.png', 'paper/fig5-models.png'];
export const PaperFigures: React.FC<SceneProps> = () => {
	const f = useCurrentFrame();
	return (
		<AbsoluteFill style={{backgroundColor: H.paper}}>
			{PLAIN_FIGURES.map((file, i) => (
				<AbsoluteFill key={file} style={{alignItems: 'center', justifyContent: 'center', opacity: interpolate(f, [i * 135, i * 135 + 12, (i + 1) * 135, (i + 1) * 135 + 12], [0, 1, 1, i === 2 ? 1 : 0], clamp)}}>
					<Img src={staticFile(file)} style={{maxWidth: 1500, maxHeight: 760}} />
				</AbsoluteFill>
			))}
		</AbsoluteFill>
	);
};
