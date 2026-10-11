// Voice: hold-to-talk (MediaRecorder → OpenAI transcription), "evera" wake word
// (webkitSpeechRecognition), and speech out (ElevenLabs, falling back to the browser voice).
// The analyser feeds the waveform for both the mic and EVERA's voice.

const WAKE = /\b(evera|ever a|evra|avera|a vera|ev era|eve era|everra|evarra|severa|e vera|vera)\b/i;
const VOCAB = 'evera, soldering iron, heat-set insert, screwdriver, multimeter, measuring tape, allen key, glue gun, pliers, tweezers, flush cutters, calipers, helping hand, return all';

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
    const type = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm'].find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || '';
    this.mime = type || 'audio/mp4';
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream, type ? { mimeType: type } : undefined);
    this.recorder.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.recorder.start(250);
    this.micSrc = this.ctx.createMediaStreamSource(this.stream);
    this.micSrc.connect(this.analyser);
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
    const done = new Promise((res) => (this.recorder.onstop = res));
    this.recorder.stop();
    await done;
    this.micSrc?.disconnect();
    this.stream.getTracks().forEach((t) => t.stop());
    this.resumeWake();
    if (held < 350) {
      this.ev.state('idle');
      return null;
    }
    this.ev.state('thinking');
    const blob = new Blob(this.chunks, { type: this.mime });
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
    if (!s.openaiKey) throw new Error('Add your OpenAI key in Settings to use hold-to-talk.');
    const ext = this.mime.includes('webm') ? 'webm' : 'mp4';
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
      if (this.awaitCmd) {
        this.ev.interim(text);
        if (res.isFinal && text) {
          clearTimeout(this.awaitCmd);
          this.awaitCmd = null;
          this.ev.command(text);
        }
        continue;
      }
      const m = text.match(WAKE);
      if (!m) continue;
      const rest = text.slice(m.index + m[0].length).replace(/^[\s,.!?]+/, '');
      if (!this.woke) {
        this.woke = true;
        this.ev.state('listening');
        this.chime();
      }
      this.ev.interim(rest || '…');
      if (res.isFinal) {
        this.woke = false;
        if (rest.split(' ').length >= 2) this.ev.command(rest);
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
