import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';

// One synth for the lifetime of the page. Notes are scheduled on Tone's
// transport so they can be cancelled; the synth itself is never disposed,
// because release callbacks for notes that are sounding can still fire
// after a stop and would throw "Synth was already disposed".
let synth = null;
let part = null;
let stopTimer = null;

const getSynth = () => {
    if (!synth) {
        synth = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.01, decay: 0.2, sustain: 0.4, release: 0.6 },
        }).toDestination();
    }
    return synth;
};

export const stopPlayback = () => {
    if (stopTimer) {
        clearTimeout(stopTimer);
        stopTimer = null;
    }
    if (part) {
        part.dispose(); // cancels every note it has not played yet
        part = null;
    }
    Tone.Transport.stop();
    Tone.Transport.cancel();
    if (synth) {
        synth.releaseAll();
    }
};

// Plays a MIDI Blob. Resolves with the duration in seconds once
// playback has been scheduled; calls onEnd when it finishes.
export const playMidiData = async (midiBlob, onEnd) => {
    // Browsers require the audio context to be started from a user gesture
    await Tone.start();
    stopPlayback();

    const midi = new Midi(await midiBlob.arrayBuffer());
    const events = [];
    midi.tracks.forEach((track) => {
        track.notes.forEach((note) => {
            events.push({
                time: note.time,
                name: note.name,
                duration: note.duration,
                velocity: note.velocity,
            });
        });
    });

    const instrument = getSynth();
    part = new Tone.Part((time, note) => {
        instrument.triggerAttackRelease(note.name, note.duration, time, note.velocity);
    }, events).start(0);
    Tone.Transport.start('+0.1');

    const duration = midi.duration;
    stopTimer = setTimeout(() => {
        stopPlayback();
        if (onEnd) onEnd();
    }, (duration + 1) * 1000);

    return duration;
};
