// Voice: hold-to-talk (raw mic samples → WAV → ElevenLabs/OpenAI transcription), "evera" wake word
// (webkitSpeechRecognition), and speech out (ElevenLabs, falling back to the browser voice).
// The analyser feeds the waveform for both the mic and EVERA's voice.

// "evera" isn't a dictionary word, so recognisers spell it many ways ("a Vera", "Avera", "Elvira"...).
// Match it loosely: any 1-3 adjacent words that, joined, are within edit distance 2 of "evera".
const WAKE_ALIASES = new Set(['evera', 'avera', 'evara', 'evra', 'elvira', 'ebira', 'abera', 'avira', 'efera', 'everra', 'evarra', 'severa', 'vera', 'heyvera', 'ivera', 'iveral', 'hevera', 'aveda', 'evita', 'evera']);
const NOT_WAKE = new Set(['every', 'ever', 'never', 'very', 'severe', 'several', 'everyone', 'everything', 'area', 'arena', 'opera', 'era', 'here', 'there', 'were', 'where', 'eva']);
function lev(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}
export function findWake(text) {
  const words = text.toLowerCase().replace(/[^a-z\s]/g, ' ').split(/\s+/).filter(Boolean);
  for (let i = 0; i < words.length; i++) {
    for (const n of [1, 2, 3]) {
      if (i + n > words.length) break;
      const cand = words.slice(i, i + n).join('');
      if (n === 1 && NOT_WAKE.has(cand)) continue;
      if (WAKE_ALIASES.has(cand) || (cand.length >= 4 && cand.length <= 8 && lev(cand, 'evera') <= 2 && !NOT_WAKE.has(cand))) {
        return { rest: words.slice(i + n).join(' ') };
      }
    }
  }
  return null;
}
const VOCAB = 'evera, soldering iron, heat-set insert, screwdriver, multimeter, measuring tape, allen key, glue gun, pliers, tweezers, flush cutters, calipers, helping hand, return all';

// Float32 sample chunks -> 16 kHz mono 16-bit WAV blob.
export function wav(chunks, inRate, outRate = 16000) {
  const len = chunks.reduce((n, c) => n + c.length, 0);
  const all = new Float32Array(len);
  let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  const ratio = inRate / outRate;
  const n = Math.floor(len / ratio);
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, outRate, true); v.setUint32(28, outRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * ratio), b = Math.min(len, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = a; j < b; j++) sum += all[j];
    const s = Math.max(-1, Math.min(1, sum / Math.max(1, b - a)));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

export class Voice {
  constructor(getSettings, ev) {
    this.getSettings = getSettings;
    this.ev = ev; // {state(s), interim(text), command(text), error(msg)}
    this.ctx = null;
    this.analyser = null;
    this.buf = new Uint8Array(256);
    this.speaking = false;
    this.recording = false;
    this.wakeOn = false;
    this.rec = null;
    this.awaitCmd = null;
  }

  get secure() {
    return window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;
  }

  get wakeSupported() {
    return this.secure && !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  // iOS only plays audio after a user gesture: call this from the first tap.
  unlock() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume();
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC();
    const b = this.ctx.createBuffer(1, 1, 22050);
    const s = this.ctx.createBufferSource();
    s.buffer = b;
    s.connect(this.ctx.destination);
    s.start(0);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.7;
  }

  // 0..1 amplitude bins for the waveform (mic while recording, voice while speaking).
  levels() {
    if (!this.analyser || !(this.recording || this.speaking)) return null;
    this.analyser.getByteFrequencyData(this.buf);
    return this.buf;
  }

  // ------------------------------------------------------------ hold to talk
  async startHold() {
    if (!this.secure) {
      this.ev.error('Voice needs HTTPS. Open EVERA through the tunnel link.');
      return false;
    }
    this.unlock();
    this.stopSpeaking();
    this.pauseWake();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch {
      this.ev.error('Microphone access was blocked.');
      this.resumeWake();
      return false;
    }
    // Capture raw samples and build a WAV ourselves. Browser recorders differ (Safari's
    // fragmented MP4 often arrives "corrupted"); a plain WAV works with every speech-to-text API.
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    this.pcm = [];
    this.micSrc = this.ctx.createMediaStreamSource(this.stream);
    this.micSrc.connect(this.analyser);
    this.proc = this.ctx.createScriptProcessor(4096, 1, 1);
    this.proc.onaudioprocess = (e) => { if (this.recording) this.pcm.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
    this.mute = this.ctx.createGain();
    this.mute.gain.value = 0; // keeps the processor running without playing the mic back
    this.micSrc.connect(this.proc);
    this.proc.connect(this.mute).connect(this.ctx.destination);
    this.recording = true;
    this.t0 = performance.now();
    this.ev.state('listening');
    this.liveCaption();
    return true;
  }

  // Live caption while holding (best effort, Web Speech).
  liveCaption() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    // iOS lets only one thing own the mic: a second recogniser can mute the recording.
    const ios = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    if (!SR || ios) return;
    try {
      const r = new SR();
      r.lang = 'en-US';
      r.interimResults = true;
      r.continuous = true;
      r.onresult = (e) => this.ev.interim([...e.results].map((x) => x[0].transcript).join(' '));
      r.onerror = () => {};
      r.start();
      this.caption = r;
    } catch { /* not available */ }
  }

  async stopHold() {
    if (!this.recording) return null;
    this.recording = false;
    try { this.caption?.stop(); } catch { /* ignore */ }
    this.caption = null;
    const held = performance.now() - this.t0;
    this.micSrc?.disconnect();
    this.proc?.disconnect();
    this.mute?.disconnect();
    this.stream.getTracks().forEach((t) => t.stop());
    this.resumeWake();
    if (held < 350) {
      this.ev.state('idle');
      return null;
    }
    this.ev.state('thinking');
    const blob = wav(this.pcm, this.ctx.sampleRate);
    try {
      return await this.transcribe(blob);
    } catch (e) {
      this.ev.error(e.message);
      this.ev.state('idle');
      return null;
    }
  }

  async transcribe(blob) {
    const s = this.getSettings();
    const ext = 'wav';
    // OpenAI transcription when the brain is OpenAI, otherwise ElevenLabs Scribe.
    if (!(s.llmProvider === 'openai' && s.openaiKey)) {
      if (!s.elevenKey) throw new Error('Add an ElevenLabs or OpenAI key to use hold-to-talk.');
      const fd = new FormData();
      fd.append('file', blob, `speech.${ext}`);
      fd.append('model_id', 'scribe_v1');
      fd.append('language_code', 'en');
      const r = await fetch('https://api.elevenlabs.io/v1/speech-to-text', { method: 'POST', headers: { 'xi-api-key': s.elevenKey }, body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.detail?.message || `Speech-to-text error ${r.status}`);
      return (await r.json()).text?.trim();
    }
    const send = async (model) => {
      const fd = new FormData();
      fd.append('file', blob, `speech.${ext}`);
      fd.append('model', model);
      fd.append('language', 'en');
      fd.append('prompt', VOCAB);
      const r = await fetch('https://api.openai.com/v1/audio/transcriptions', { method: 'POST', headers: { Authorization: `Bearer ${s.openaiKey}` }, body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.error?.message || `Transcription error ${r.status}`);
      return (await r.json()).text?.trim();
    };
    const model = s.sttModel || 'gpt-4o-mini-transcribe';
    try {
      return await send(model);
    } catch (e) {
      if (model !== 'whisper-1') return send('whisper-1');
      throw e;
    }
  }

  // ------------------------------------------------------------ wake word
  setWake(on) {
    this.wakeOn = on;
    if (on) this.resumeWake();
    else this.pauseWake();
  }

  resumeWake() {
    if (!this.wakeOn || !this.wakeSupported || this.rec || this.recording) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const r = new SR();
    r.lang = 'en-US';
    r.continuous = true;
    r.interimResults = true;
    r.onresult = (e) => this.onWakeResult(e);
    r.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        this.wakeOn = false;
        this.ev.error('Wake word needs microphone and speech permission.');
      }
    };
    r.onend = () => {
      this.rec = null;
      if (this.wakeOn && !this.recording) setTimeout(() => this.resumeWake(), 250);
    };
    try {
      r.start();
      this.rec = r;
    } catch { /* already running */ }
  }

  pauseWake() {
    const r = this.rec;
    this.rec = null;
    if (r) {
      r.onend = null;
      try { r.abort(); } catch { /* ignore */ }
    }
  }

  onWakeResult(e) {
    if (this.speaking) return; // don't hear ourselves
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i];
      const text = res[0].transcript.trim();
      this.ev.heard?.(text);
      if (this.awaitCmd) {
        this.ev.interim(text);
        if (res.isFinal && text) {
          clearTimeout(this.awaitCmd);
          this.awaitCmd = null;
          this.ev.command(text);
        }
        continue;
      }
      const m = findWake(text);
      if (!m) continue;
      const rest = m.rest;
      if (!this.woke) {
        this.woke = true;
        this.ev.state('listening');
        this.chime();
      }
      this.ev.interim(rest || '…');
      if (res.isFinal) {
        this.woke = false;
        if (rest.length >= 3) this.ev.command(rest);
        else {
          this.awaitCmd = setTimeout(() => {
            this.awaitCmd = null;
            this.ev.state('idle');
          }, 7000);
        }
      }
    }
  }

  chime() {
    if (!this.ctx) return;
    const o = this.ctx.createOscillator(), g = this.ctx.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(880, this.ctx.currentTime);
    o.frequency.exponentialRampToValueAtTime(1320, this.ctx.currentTime + 0.12);
    g.gain.setValueAtTime(0.0001, this.ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.12, this.ctx.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.3);
    o.connect(g).connect(this.ctx.destination);
    o.start();
    o.stop(this.ctx.currentTime + 0.32);
  }

  // ------------------------------------------------------------ speech out
  async speak(text, onStart) {
    if (!text) return;
    this.stopSpeaking();
    const s = this.getSettings();
    const id = (this.speakId = Math.random());
    if (s.elevenKey && this.ctx) {
      try {
        const voice = s.elevenVoice || 'JBFqnCBsd6RMkjVDRZzb';
        const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voice)}?output_format=mp3_44100_128`, {
          method: 'POST',
          headers: { 'xi-api-key': s.elevenKey, 'Content-Type': 'application/json', Accept: 'audio/mpeg' },
          body: JSON.stringify({ text, model_id: s.elevenModel || 'eleven_flash_v2_5', voice_settings: { stability: 0.45, similarity_boost: 0.8, style: 0.25 } }),
        });
        if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.detail?.message || `ElevenLabs error ${r.status}`);
        const audio = await this.ctx.decodeAudioData(await r.arrayBuffer());
        if (id !== this.speakId) return;
        const src = this.ctx.createBufferSource();
        src.buffer = audio;
        src.connect(this.analyser);
        this.analyser.connect(this.ctx.destination);
        this.src = src;
        this.speaking = true;
        this.ev.state('speaking');
        onStart?.(audio.duration);
        await new Promise((res) => {
          src.onended = res;
          src.start();
        });
        if (id === this.speakId) this.endSpeak();
        return;
      } catch (e) {
        this.ev.error(e.message);
      }
    }
    await this.browserSpeak(text, id, onStart);
  }

  browserSpeak(text, id, onStart) {
    return new Promise((res) => {
      if (!window.speechSynthesis) return res();
      const u = new SpeechSynthesisUtterance(text);
      const v = speechSynthesis.getVoices();
      u.voice = v.find((x) => /en-GB/i.test(x.lang) && /(Daniel|Arthur|male)/i.test(x.name)) || v.find((x) => /en-GB/i.test(x.lang)) || null;
      u.rate = 1.04;
      u.onstart = () => {
        this.speaking = true;
        this.ev.state('speaking');
        onStart?.(text.length / 14);
      };
      u.onend = u.onerror = () => {
        if (id === this.speakId) this.endSpeak();
        res();
      };
      speechSynthesis.speak(u);
    });
  }

  endSpeak() {
    this.speaking = false;
    try { this.analyser?.disconnect(this.ctx.destination); } catch { /* not connected */ }
    this.ev.state('idle');
  }

  stopSpeaking() {
    this.speakId = Math.random();
    try { this.src?.stop(); } catch { /* ignore */ }
    this.src = null;
    if (window.speechSynthesis) speechSynthesis.cancel();
    if (this.speaking) this.endSpeak();
  }
}
