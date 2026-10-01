import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';

let synth = null;
let stopTimer = null;

export const stopPlayback = () => {
    if (stopTimer) {
        clearTimeout(stopTimer);
        stopTimer = null;
    }
    if (synth) {
        synth.releaseAll();
        synth.dispose();
        synth = null;
    }
};

// Plays a MIDI Blob. Resolves with the duration in seconds once
// playback has been scheduled; calls onEnd when it finishes.
export const playMidiData = async (midiBlob, onEnd) => {
    // Browsers require the audio context to be started from a user gesture
    await Tone.start();
    stopPlayback();

    const midi = new Midi(await midiBlob.arrayBuffer());

    synth = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'triangle' },
        envelope: { attack: 0.01, decay: 0.2, sustain: 0.4, release: 0.6 },
    }).toDestination();

    const now = Tone.now() + 0.1;
    midi.tracks.forEach((track) => {
        track.notes.forEach((note) => {
            synth.triggerAttackRelease(note.name, note.duration, now + note.time, note.velocity);
        });
    });

    const duration = midi.duration;
    stopTimer = setTimeout(() => {
        stopPlayback();
        if (onEnd) onEnd();
    }, (duration + 1) * 1000);

    return duration;
};
