import React from 'react';
import MusicGenerator from './components/MusicGenerator';
import './styles/App.css';

function App() {
    return (
        <div className="App">
            <header className="App-header">
                <h1>🎵 AI Music Composer</h1>
                <p>Generate beautiful melodies with artificial intelligence</p>
            </header>

            <main className="App-main">
                <MusicGenerator />
            </main>

            <footer className="App-footer">
                <p>Built with React, PyTorch, and ❤️</p>
            </footer>
        </div>
    );
}

export default App;