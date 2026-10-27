# Build Plan: Chapter 01 — WebGL 2 Demos (Computeranimation)

> **Audience of this file: Claude, in a fresh session with no prior context.**
> Read the whole file before writing any code. It is a complete specification. Where it is precise (ids, filenames, shader interfaces, header/footer markup), follow it exactly. Where it leaves freedom, choose the simplest solution that satisfies the acceptance criteria.

---

## 0. How this file is used

**The user will upload with this file:**
1. `00_computeranimation_intro.ipynb`: a reference notebook from this course (Computeranimation, Chapter 00). Use it **only** as a visual and structural reference for the header/footer markup, pill link style, and info-box style. Do **not** copy its content. The demos are standalone dark-themed HTML files, not notebooks.

If the reference file is missing, everything needed is embedded in Section 3. Say which file is missing and continue.

**Task:** create **three standalone HTML demo files** (Section 5), one at a time. After each file, stop, present it, and wait for the user's feedback before building the next one, unless the user says "build all".

**Do not** create exercises, tasks or "try it yourself" sections. The user explicitly does not want them.

---

## 1. Project context

- **Course:** *Computeranimation*, bachelor's level, Multimedia Engineering.
- **Institute:** Institut für Medientechnik – Technische Universität Ilmenau (AMS Group).
- **Format:** 90-minute seminars. This is Chapter 01 of the Computeranimation seminar series.
- **Instructor:** Renato Fera P. (credits appear in header and footer).
- **Students:** know basic JavaScript and have completed a WebGL fundamentals chapter (they know the rendering pipeline, MVP matrices, buffers, VAOs, uniforms, and the animation loop with `requestAnimationFrame`). They see animation *principles* for the first time.
- **Pedagogy:** The demos are **interactive, tutorial-style code** that students read next to the running page. They build on each other and are worked through in order. The demos are **fully independent** from the Multimediaprogrammierung WebGL demos — no shared files, no cross-references.
- **Repository:** https://github.com/TUIlmenauAMS/multimedia_programming_seminars (files live in the repo root).
- **Language:** all page text and code comments in **English**. Credits keep their German titles ("Institut für Medientechnik", "Fachgebietsleiter", "Wissenschaftlicher Mitarbeiter").

---

## 2. Global constraints and conventions

### 2.1 Technology
- **WebGL 2 only** (`canvas.getContext('webgl2')`), shaders in **GLSL ES 3.00** (`#version 300 es` as the very first characters of the source string).
- **Raw WebGL. No libraries**, no CDN scripts, no external fonts, no external images. Each file is **fully self-contained** (one `.html` file with inline `<style>` and `<script>`), and works when opened via `file://`.
- The only external resources allowed are the AMS logo `<img>` from the template (raw.githubusercontent URL) and outbound `<a>` links.
- No `localStorage`/`sessionStorage`; no build step; no frameworks.
- Target: current Chrome, Edge, Firefox, Safari.

### 2.2 Professional code standards (JavaScript)
- A single `<script type="module">` at the end of `<body>` (strict mode by default). No globals.
- `const` by default, `let` only when reassigned. No `var`.
- Small, single-purpose functions with **JSDoc** (`@param`, `@returns`).
- Constants for magic numbers (`const DEFAULT_DURATION = 2.0;`).
- Throw/handle errors: always check `getContext`, shader compile status and program link status, and print the info log.
- Show a friendly in-page error banner (`.error-banner`) if WebGL 2 is unavailable or a shader fails; never leave a silent black canvas.
- Handle `devicePixelRatio` and resizing (`resizeCanvasToDisplaySize`). Redraw on `resize`.
- Use `requestAnimationFrame` with the timestamp argument. Never use `setInterval` for rendering.
- Use performance-independent animation: derive positions from accumulated time, not frame counts.
- Cache all `getUniformLocation` / `getAttribLocation` results at init. Never look them up per frame.
- Initialization happens once; the render function only sets uniforms, binds the VAO and draws.
- Event listeners via `addEventListener`; DOM lookups via `document.querySelector`/`getElementById`, collected once in a small `ui` object.

### 2.3 Professional code standards (CSS)
- Use CSS custom properties for the palette (define `:root` variables).
- Mobile-friendly: flex/grid, `max-width: 100%`, no horizontal page scroll.
- Class names are lowercase-kebab (`control-row`, `gl-canvas`).

### 2.4 Tutorial-style comments (most important requirement)
The source is meant to be **read by students**. Therefore:
- Open the script with a **block comment** stating: demo title, what it teaches, the key animation concepts covered, and the list of steps in the file.
- Divide the script into numbered sections with banner comments:
  ```js
  // ══════════════════════════════════════════════════════════════
  // STEP 3 — Build the keyframe track
  // ══════════════════════════════════════════════════════════════
  ```
- Before every non-trivial block, a short comment explains **what** it does and **why** (not just a restatement of the code line).
- Where a helper is standard WebGL boilerplate (createShader, createProgram, resizeCanvasToDisplaySize), mark it `// Standard WebGL helper — see the Multimediaprogrammierung WebGL chapter for a full walkthrough.` and keep the comment brief.
- Where something is the **animation concept being taught**, mark it `// ANIMATION CONCEPT:` and give a thorough explanation.
- Comment each GLSL line group.
- Do not over-comment trivial lines. Aim for clarity, not noise.

### 2.5 Page naming and IDs
- Files (exactly these names):
  1. `01_ca_keyframes.html`
  2. `01_ca_easing.html`
  3. `01_ca_camera.html`
- Canvas id is always `glcanvas`. Controls use the ids listed in each demo spec.

### 2.6 Naming in shaders
`a_` attributes, `u_` uniforms, `v_` varyings. Fragment output is always `outColor`.

---

## 3. Page template

Every demo page has this structure:

1. **Header** (logo, title, institute, credits). Exact markup in 3.1.
2. **Section 1 — Concept** (`h2.section`): 1–3 short paragraphs introducing the animation principle, and an `.info-box` with the key formula or concept summary.
3. `<hr />`
4. **Section 2 — Interactive Demo** (`h2.section`): one sentence, then `.demo-card` containing the `<canvas id="glcanvas" class="gl-canvas">`, the controls, and a small `.stats` read-out.
5. `<hr />`
6. **Section 3 — How It Works**: explains the code structure, referencing the step names in the source. Includes an `.info-box` with a note.
7. **Pills** with links (MDN, relevant animation resources).
8. **Footer**.
9. `<script type="module">`.

Keep prose short. The code comments carry the teaching.

### 3.1 Header and footer markup (use exactly, change only the demo name)

```html
<header class="nb-header">
  <img
    src="https://raw.githubusercontent.com/TUIlmenauAMS/multimedia_programming_seminars/main/ams_group_tui.png"
    alt="AMS Group – TU Ilmenau"
  />
  <div>
    <h1>Computeranimation</h1>
    <h2>Chapter 01: Animation Fundamentals &mdash; Demo: DEMO_NAME</h2>
    <p><strong>Institut für Medientechnik &ndash; Technische Universität Ilmenau</strong><br>
      <a href="https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-elektrotechnik-und-informationstechnik/profil/institute-und-fachgebiete/fachgebiet-angewandte-mediensysteme" target="_blank">
        Institut für Medientechnik &ndash; AMS Group
      </a>
    </p>
    <p>
      <strong>Fachgebietsleiter:</strong> Prof. Dr.-Ing. Gerald Schuller<br>
      <strong>Wissenschaftlicher Mitarbeiter:</strong> Renato Fera P.
      <a href="https://www.linkedin.com/in/renato-profeta/" target="_blank">LinkedIn</a>
    </p>
  </div>
</header>
```

```html
<footer class="nb-footer">
  <p><strong>Computeranimation</strong> &middot; Chapter 01: Animation Fundamentals &middot; Demo: DEMO_NAME<br>
    Institut für Medientechnik &ndash; Technische Universität Ilmenau</p>
  <p>Demo created by <strong>Renato Fera P.</strong> &middot;
    <a href="https://www.linkedin.com/in/renato-profeta/" target="_blank">LinkedIn</a>
  </p>
</footer>
```

`DEMO_NAME` values: Keyframes & Interpolation, Easing & Animation Curves, Camera Animation.

### 3.2 Base CSS

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --accent: #1a73e8;
  --danger: #e53935;
  --border: #e0e0e0;
  --canvas-bg: #1e1e2e;
  --surface: #f8f9fa;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 16px; line-height: 1.6; color: #212121; background: #fff;
  max-width: 900px; margin: 0 auto; padding: 0 24px 48px;
}

.nb-header { display: flex; align-items: center; gap: 24px; border-bottom: 2px solid var(--border); padding: 16px 0; margin-bottom: 32px; }
.nb-header img { width: 120px; border-radius: 12px; flex-shrink: 0; }
.nb-header h1 { font-size: 1.6rem; margin-bottom: 4px; }
.nb-header h2 { font-size: 1.1rem; font-weight: 400; margin-bottom: 8px; color: #444; }
.nb-header p  { font-size: 0.92rem; margin-bottom: 2px; }
.nb-header a  { color: var(--accent); text-decoration: none; }
.nb-header a:hover { text-decoration: underline; }

.pills { margin: 12px 0; display: flex; flex-wrap: wrap; gap: 8px; }
.pill { display: inline-block; padding: 6px 14px; border: 1px solid var(--accent); border-radius: 20px;
  text-decoration: none; font-weight: 600; font-size: 0.92rem; color: var(--accent); transition: background 0.15s, color 0.15s; }
.pill:hover { background: var(--accent); color: #fff; }

h2.section { font-size: 1.3rem; margin: 40px 0 12px; }
hr { border: none; border-top: 1px solid var(--border); margin: 36px 0; }
p  { margin-bottom: 12px; }

.demo-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px 32px; margin: 24px 0; }
.demo-card h3 { font-size: 1rem; font-weight: 600; margin-bottom: 20px; color: #444; text-transform: uppercase; letter-spacing: 0.04em; }

.control-row { display: flex; align-items: center; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }
.control-row label { min-width: 140px; font-weight: 500; font-size: 0.95rem; }
.control-row input[type=range] { flex: 1; min-width: 160px; accent-color: var(--accent); }
.control-row select { padding: 6px 10px; border: 1px solid #ccc; border-radius: 8px; font-size: 0.95rem; background: #fff; }
.value-badge { min-width: 64px; text-align: right; font-variant-numeric: tabular-nums; font-size: 0.92rem; color: #555; }

.btn-row { display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.btn { padding: 10px 28px; border: none; border-radius: 24px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: opacity 0.15s, transform 0.1s; }
.btn:active { transform: scale(0.97); }
.btn-primary { background: var(--accent); color: #fff; }
.btn-danger  { background: var(--danger); color: #fff; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }

canvas.gl-canvas { display: block; width: 100%; height: auto; aspect-ratio: 16 / 9; border-radius: 8px; background: var(--canvas-bg); margin-top: 20px; touch-action: none; }
canvas.gl-canvas.square { aspect-ratio: 1 / 1; max-width: 520px; margin-left: auto; margin-right: auto; }

.canvas-wrap { margin-bottom: 16px; }
.stats { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.82rem; color: #555; margin-top: 10px; min-height: 1.4em; }
.info-box { border-left: 5px solid var(--accent); background: rgba(26,115,232,0.07); border-radius: 6px; padding: 12px 18px; margin: 20px 0; font-size: 0.95rem; }
.error-banner { background: #fdecea; border-left: 5px solid var(--danger); border-radius: 6px; padding: 12px 18px; margin: 20px 0; color: #7f1d1d; }
code { background: #f1f3f4; border-radius: 4px; padding: 1px 6px; font-size: 0.9em; font-family: "SFMono-Regular", Consolas, monospace; }

.nb-footer { text-align: center; padding: 18px 0 8px; margin-top: 48px; border-top: 2px solid var(--border); font-size: 0.92rem; color: #444; line-height: 1.7; }
.nb-footer a { color: var(--accent); text-decoration: none; }
.nb-footer a:hover { text-decoration: underline; }

/* 2D overlay canvas (demos 1 & 2: animation curve plot) */
canvas.curve-canvas { display: block; width: 100%; height: auto; aspect-ratio: 2 / 1; border-radius: 8px; background: #fafafa; border: 1px solid var(--border); margin-top: 12px; }

@media (max-width: 600px) {
  .nb-header { flex-direction: column; align-items: flex-start; }
  .demo-card { padding: 20px 16px; }
}
```

### 3.3 Script skeleton (every demo follows this order)

```js
/**
 * ═══════════════════════════════════════════════════════════════
 * Computeranimation — Chapter 01 · Demo: <n>
 * Animation concepts: <list>.
 * Steps in this file: 1) … 2) … 3) …
 * ═══════════════════════════════════════════════════════════════
 */

// STEP 0 — Collect DOM references once
// STEP 1 — Get the WebGL 2 context (with friendly error banner)
// STEP 2 — Shader sources (GLSL ES 3.00)
// STEP 3 — Standard WebGL helpers: createShader, createProgram, resizeCanvasToDisplaySize
// STEP 4 — Compile + link, cache attribute/uniform locations
// STEP 5 — Build geometry buffers and VAO(s)
// STEP 6 — Animation state and keyframe/curve data
// STEP 7 — Wire UI events to state
// STEP 8 — Render + animation update function
// STEP 9 — Start the loop
```

### 3.4 Shared WebGL helpers (copy identically into each file)

```js
/** Standard WebGL helper — see Chapter 01 of the WebGL seminar for a full walkthrough. */
function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error:\n${log}`);
  }
  return shader;
}

function createProgram(gl, vertSrc, fragSrc) {
  const vs = createShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error:\n${log}`);
  }
  return prog;
}

function resizeCanvasToDisplaySize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h; return true;
  }
  return false;
}
```

### 3.5 mat4 helpers (column-major Float32Array; hand-written, no gl-matrix)

```js
// ANIMATION CONCEPT: All 3D transforms are 4×4 matrices in column-major order.
// These are the same helpers as in the WebGL chapter — included here for self-containment.
const mat4 = {
  identity:    () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  translation: (x,y,z) => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]),
  rotationX:   (a) => { const c=Math.cos(a),s=Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); },
  rotationY:   (a) => { const c=Math.cos(a),s=Math.sin(a); return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); },
  rotationZ:   (a) => { const c=Math.cos(a),s=Math.sin(a); return new Float32Array([c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1]); },
  scaling:     (x,y,z) => new Float32Array([x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1]),
  perspective: (fovY, aspect, near, far) => {
    const f = 1/Math.tan(fovY/2), nf = 1/(near-far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
  },
  multiply: (a, b) => {
    const out = new Float32Array(16);
    for (let c=0;c<4;c++) for (let r=0;r<4;r++) {
      let s=0; for (let k=0;k<4;k++) s += a[k*4+r]*b[c*4+k];
      out[c*4+r]=s;
    }
    return out;
  },
  /** Look-at view matrix: eye position, target, up vector (all vec3 arrays). */
  lookAt: (eye, target, up) => {
    // ANIMATION CONCEPT: lookAt constructs the view matrix from camera position and target.
    // The three axes of the camera's local frame are computed from cross products.
    const normalize = v => { const l=Math.hypot(...v); return v.map(x=>x/l); };
    const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    const dot   = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    const z = normalize([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]);
    const x = normalize(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -dot(x,eye), -dot(y,eye), -dot(z,eye), 1,
    ]);
  },
};
```

---

## 4. Teaching progression

| # | Demo | New animation concept | Builds on |
|---|------|-----------------------|-----------|
| 1 | Keyframes & Interpolation | Keyframe tracks, linear lerp, why constant velocity looks mechanical | WebGL rAF loop, uniforms |
| 2 | Easing & Animation Curves | Easing functions (ease-in/out, cubic Bézier), live curve plot, timing vs. spacing | Demo 1 keyframe idea |
| 3 | Camera Animation | View matrix as animation target, orbit path, dolly, look-at interpolation | Demo 2 easing, mat4.lookAt |

Each demo's code should be as short as possible so the **new** animation concept stands out. Mark reused boilerplate briefly and mark new animation logic with `// ANIMATION CONCEPT:`.

---

## 5. Demo specifications

### 5.1 `01_ca_keyframes.html` — Keyframes & Interpolation

**Goal:** Introduce the keyframe as the fundamental unit of computer animation. Show how values are stored at discrete times and interpolated between them. Make the difference between linear interpolation and stepped (no interpolation) visible and tactile.

**Scene:** a colored sphere (UV sphere, ~24×16 segments, smooth normals) moving along a path in a simple 3D scene. A ground plane (flat quad, a neutral grey) gives spatial reference. One directional light (same Lambert model as in the WebGL chapter, reused without explanation). Camera is fixed: `eye = [0, 3, 8]`, target = `[0, 0.5, 0]`.

**Keyframe track:** a JavaScript array of keyframes, each `{ time: number, position: [x,y,z], color: [r,g,b] }`. Default track (5 keyframes over 4 seconds):

| time | position | color |
|------|----------|-------|
| 0.0 | [−3, 0.5, 0] | [0.9, 0.3, 0.2] red |
| 1.0 | [−1, 1.5, 1] | [0.9, 0.7, 0.1] orange |
| 2.0 | [ 0, 0.5, 0] | [0.2, 0.8, 0.4] green |
| 3.0 | [ 1, 2.0, −1] | [0.2, 0.5, 0.9] blue |
| 4.0 | [ 3, 0.5, 0] | [0.7, 0.2, 0.9] purple |

**Interpolation modes** (`interpMode` select):
- `linear` — lerp between adjacent keyframes: `value = a + (b − a) * t` where `t = (time − kf[i].time) / (kf[i+1].time − kf[i].time)`. Mark this formula with `// ANIMATION CONCEPT:` and explain what `t` represents (normalized local time, 0 at the start keyframe, 1 at the end).
- `step` — hold the value of the previous keyframe until the next one is reached (no interpolation). Used in sprite animation; shows why interpolation matters.

Both position and color are interpolated (or stepped) using the same `t`.

**Playback:** the animation loops. Accumulate `playTime` each frame only when not paused. When `playTime > totalDuration` wrap it around (`playTime %= totalDuration`).

**Scrubber:** a range input (`scrubber`, 0–1, step 0.001) that sets `playTime = value * totalDuration` directly. When the user drags the scrubber, set `paused = true` automatically. A `play` button resumes. This teaches students that an animation is just a function of time — time can be driven by anything.

**Keyframe markers:** draw the 5 keyframe positions as small flat discs on the ground plane (using `gl.POINTS` with `gl_PointSize` or tiny quad billboards — whichever is simpler). They stay fixed; the sphere moves between them. Explain in a comment why this helps understand the animation data.

**Controls:** `interpMode` (select: linear / step), `speed` (range 0.25–3, step 0.05, default 1.0), `scrubber` (range 0–1), `play` (button, label toggles Play/Pause), `loop` (checkbox, default on; when off, stop at end).

**Stats line:** `t = 2.34 s / 4.00 s · segment: 2→3 · mode: linear · FPS: 60`.

**Info box (Section 1):** "A keyframe stores the value of a property at a specific point in time. The animator sets the keyframes; the computer fills in the frames between them — this is called *inbetweening* or *interpolation*. Linear interpolation is the simplest: it assumes constant velocity between keyframes."

**Section 3 note:** Point out that interpolating color with lerp in RGB is not perceptually uniform (mention HSL or Oklab as alternatives used in professional tools) but RGB lerp is used here for simplicity.

**Pills:** Keyframe – Wikipedia, Linear interpolation – MDN (`CSS` lerp), Animation principles (Disney 12 principles link or Wikipedia).

**Acceptance:** sphere moves smoothly between the 5 positions and changes color in `linear` mode; in `step` mode it jumps. Scrubbing to any position shows the correct interpolated state. Speed slider scales playback rate. Keyframe markers are visible. No animation jump on resume after scrub.

---

### 5.2 `01_ca_easing.html` — Easing & Animation Curves

**Goal:** Show that *when* something moves matters as much as *where* it moves. Introduce easing functions as a mapping from linear time `t ∈ [0,1]` to eased time `e ∈ [0,1]`, and show the curve of that mapping live next to the animation.

**Scene:** the same sphere from Demo 1 (reuse the geometry and shaders, marking them as repeated). The sphere moves along a single straight-line segment between two fixed points: `from = [−3, 0.5, 0]`, `to = [3, 0.5, 0]`. The motion repeats (ping-pong: go, come back). One fixed directional light. Same fixed camera. Keep the scene minimal so the focus is entirely on timing.

**Easing functions** (`easingFn` select). Implement each as a pure JavaScript function `f(t) → number` where input and output are both in `[0,1]`. Clamp input to `[0,1]` in a wrapper. Group them clearly with banner comments:

```
// ── Linear ──────────────────────────────────────────────────
linear:           t => t

// ── Quadratic ───────────────────────────────────────────────
easeInQuad:       t => t * t
easeOutQuad:      t => t * (2 - t)
easeInOutQuad:    t => t < 0.5 ? 2*t*t : -1 + (4 - 2*t)*t

// ── Cubic ───────────────────────────────────────────────────
easeInCubic:      t => t * t * t
easeOutCubic:     t => (--t) * t * t + 1
easeInOutCubic:   t => t < 0.5 ? 4*t*t*t : (t-1)*(2*t-2)*(2*t-2)+1

// ── Cubic Bézier (custom) ────────────────────────────────────
// ANIMATION CONCEPT: CSS transition-timing-function and After Effects use
// cubic Bézier curves defined by two control points P1 and P2 (P0=0,0 P3=1,1).
// The x axis is input time; the y axis is the eased value.
// We solve for t numerically (Newton's method, 4 iterations is enough).
cubicBezier(x1, y1, x2, y2):  // implemented as a factory function → f(t)
```

For the cubic Bézier implementation use the standard CSS cubic-bezier approach: given control points `(x1,y1)` and `(x2,y2)`, solve `B_x(τ) = t` for `τ` with 4 Newton iterations, then return `B_y(τ)`. Explain the algorithm with `// ANIMATION CONCEPT:` comments. Presets for the cubic Bézier picker:

| Preset name | x1 | y1 | x2 | y2 | What it looks like |
|-------------|----|----|----|----|-------------------|
| ease (CSS default) | 0.25 | 0.1 | 0.25 | 1.0 | gentle ease-in, quick out |
| ease-in | 0.42 | 0.0 | 1.0 | 1.0 | slow start |
| ease-out | 0.0 | 0.0 | 0.58 | 1.0 | slow end |
| ease-in-out | 0.42 | 0.0 | 0.58 | 1.0 | slow start and end |
| custom | (from sliders) | | | | |

**Live curve plot (2D canvas overlay, NOT WebGL):** a `<canvas class="curve-canvas">` below the main canvas, drawn with the 2D Canvas API. Every frame (or on parameter change):
- Draw a grey grid (lines at 0.25 intervals).
- Draw the axes (x = input time t, y = output eased value e).
- Draw the easing curve as a polyline (sample 200 points from `t=0` to `t=1`).
- Draw a moving dot on the curve at the current `t` value of the animation. This dot is the most important element — it shows the student exactly where on the curve the animation currently is.
- For cubic Bézier mode: draw the two control point handles (lines from (0,0) to P1 and from (1,1) to P2, with draggable dots).

The curve plot must be **interactive for cubic Bézier**: the user can drag P1 and P2 within `[0,1]²` with pointer events on the 2D canvas. Dragging updates the easing function immediately.

**Controls:** `easingFn` (select: all options above, grouped with `<optgroup>`), `bezierPreset` (select: visible only when `cubicBezier` is selected), `x1`,`y1`,`x2`,`y2` sliders (range 0–1, step 0.01; visible only when `cubicBezier` + `custom` preset), `speed` (range 0.25–2, default 1), `paused` (checkbox).

**Stats line:** `t = 0.72 · eased = 0.91 · fn: easeOutCubic · FPS: 60`.

**Info box (Section 1):** "Easing is the difference between motion that *looks right* and motion that looks mechanical. In nature, objects accelerate and decelerate — they do not move at constant speed. An easing function remaps the linear progress `t ∈ [0,1]` to a non-linear output `e ∈ [0,1]`. The same easing function is used in CSS transitions, game engines, and professional animation software."

**Section 3 note:** mention that after Effects uses the same cubic Bézier model; the `cubic-bezier()` CSS function takes the same four parameters; `GSAP`'s ease strings compile to the same math.

**Pills:** cubic-bezier – MDN, CSS easing functions – MDN, [cubic-bezier.com](https://cubic-bezier.com) (interactive reference), Robert Penner easing equations (original source).

**Acceptance:** all easing functions produce visibly different motion. The live dot on the curve moves in sync with the sphere. Dragging P1/P2 in cubic Bézier mode updates both the sphere and the curve in real time. `linear` produces a diagonal line. `easeInOutCubic` produces the classic S-curve.

---

### 5.3 `01_ca_camera.html` — Camera Animation

**Goal:** Show that the camera is just another animated object: its position, target, and field of view are values that change over time. Demonstrate orbit, dolly, and look-at interpolation as the three most common camera animation patterns.

**Scene:** a small stage with 3–4 objects at fixed positions — use simple geometry to keep the scene interesting without complicating the code:
- A **cube** at `[−1.5, 0.5, 0]` (flat-shaded, colored faces, as in the WebGL chapter).
- A **UV sphere** at `[1.5, 0.5, 0]` (smooth-shaded, single color).
- A **ground plane** (flat quad, grey).
One directional light, world-space, fixed. Mark all geometry helpers `// Standard geometry helper — see the Multimediaprogrammierung WebGL demos for a full walkthrough.`

**Camera modes** (`cameraMode` select):

**`orbit`** — the camera moves along a circular arc around a fixed target point (`[0, 0.5, 0]`). The eye position is:
```
eye.x = target.x + radius * sin(azimuth) * cos(elevation)
eye.y = target.y + radius * sin(elevation)
eye.z = target.z + radius * cos(azimuth) * cos(elevation)
```
`azimuth` increases over time (auto-rotate) or is controlled by a slider. `elevation` and `radius` are sliders. Explain the spherical coordinate formula with `// ANIMATION CONCEPT:`.

**`dolly`** — the camera moves along a straight line path defined by two keyframes (start position and end position), looking at the fixed target `[0, 0.5, 0]` throughout. The path is a lerp between `eye_start = [0, 2, 8]` and `eye_end = [0, 1, 2]`. Time is driven by the `scrubber` slider or by auto-play. Apply the easing function selected by `dollyCurve` (select: linear / easeInOut / easeOutCubic) to the scrubber value before lerping. This connects Demo 2 directly.

**`lookAtLerp`** — two cameras (A and B) are defined with their own `eye` position and `target` point. The rendered camera interpolates between them: `lerpedEye = lerp(A.eye, B.eye, t)`, `lerpedTarget = lerp(A.target, B.target, t)`. Show a "cut" button (instant switch, `t` snaps to 0 or 1) and a "blend" button (smooth transition using easeInOutCubic over 1.5 seconds). Explain `// ANIMATION CONCEPT: interpolating both eye and target produces a camera move; interpolating neither is a cut; interpolating only t between 0 and 1 is a blend.`

Presets for camera A and B (label them in the UI):
- A: `eye=[−4, 2, 4]`, `target=[0, 0.5, 0]` — wide establishing shot
- B: `eye=[0.5, 1, 2]`, `target=[1.5, 0.5, 0]` — close-up on the sphere

**FOV animation:** in all three modes, a `fovAnim` checkbox (default off) makes the field of view oscillate between 30° and 70° with a slow sine wave (period 6 s). Explain `// ANIMATION CONCEPT: animating FOV simulates a zoom lens ("dolly zoom" effect when combined with a dolly move).`

**Controls (orbit mode):** `azimuth` (range 0–360, auto-increment when `autoRotate` is on), `elevation` (range 5–80°, default 25), `radius` (range 2–10, default 6), `autoRotate` (checkbox, default on), `rotSpeed` (range 0.1–2, default 0.5).

**Controls (dolly mode):** `scrubber` (range 0–1), `dollyCurve` (select), `play` button.

**Controls (lookAtLerp mode):** `cut` button, `blend` button, `blendProgress` read-only badge showing current `t`.

**Controls (all modes):** `cameraMode` (select), `fovAnim` (checkbox), `fov` (range 20–90, default 45, disabled when `fovAnim` is on).

**Stats line (orbit):** `eye: (−3.2, 1.8, 4.7) · target: (0, 0.5, 0) · fov: 45° · FPS: 60`.
**Stats line (dolly):** `t = 0.62 · easing: easeOutCubic · eye: (0.0, 1.4, 3.3) · FPS: 60`.
**Stats line (lookAtLerp):** `blend: 0.74 · eye: (−1.1, 1.4, 2.8) · FPS: 60`.

**Info box (Section 1):** "In computer animation, the camera is not special — it is just another object with a position and orientation that can be keyframed and interpolated. The `lookAt` view matrix takes an eye position, a target point, and an up vector and constructs the transformation that maps world space to camera space. Animating these three values over time is all that is needed to produce a camera move."

**Section 3 note:** mention that professional DCCs (Blender, Maya) expose exactly these same parameters on their camera objects. The orbit control is the same math used in Three.js `OrbitControls`. The cubic Bézier easing from Demo 2 can be dropped in here with zero changes.

**Pills:** `mat4.lookAt` explained – learnopengl.com, Camera (Blender manual), Three.js OrbitControls source (GitHub, for reference).

**Acceptance:** orbit mode shows continuous smooth rotation around the stage; elevation and radius sliders change the viewpoint visibly. Dolly mode moves the camera from far to close; different `dollyCurve` settings produce noticeably different timing. LookAt-lerp "cut" instantly switches; "blend" smoothly transitions. FOV animation produces a visible zoom effect. `mat4.lookAt` is used for all three modes (not matrix stacking).

---

## 6. Demo index block (for the theory notebook, if one exists)

If the user provides a theory notebook for this chapter, each demo should be introduced right after the section it illustrates, using this format:

```html
<div style="border-left:5px solid #1a73e8; background:rgba(26,115,232,0.07); border-radius:6px; padding:12px 18px; margin:20px 0;">
  <p style="margin:0 0 6px 0;"><b>Interactive Demo N &mdash; TITLE</b></p>
  <p style="margin:0 0 6px 0;">One or two sentences: what the demo shows.</p>
  <p style="margin:0 0 6px 0;"><i>What to look for:</i> one concrete thing the student should try.</p>
  <a href="https://htmlpreview.github.io/?url=https://raw.githubusercontent.com/TUIlmenauAMS/multimedia_programming_seminars/main/FILE.html" target="_blank" style="display:inline-block; padding:6px 14px; margin:4px 8px 4px 0; border:1px solid #1a73e8; border-radius:20px; text-decoration:none; font-weight:600; font-size:0.95em;">Open TITLE Demo &#8599;</a>
</div>
```

| Demo | What to look for |
|-----:|-----------------|
| 1 | Switch between linear and step interpolation; drag the scrubber to any point in time and observe the sphere's position and color. |
| 2 | Select `easeInOutCubic` and watch the S-curve on the plot; drag the P1/P2 handles in cubic Bézier mode and see the sphere respond immediately. |
| 3 | In orbit mode, change elevation and radius; in dolly mode, compare linear vs. easeOutCubic; use the blend button in lookAtLerp mode and watch `t` count up. |

---

## 7. Quality gates (run before presenting each file)

1. **Syntax:** extract the `<script type="module">` body, save as `.mjs`, run `node --check`.
2. **Pure-logic unit tests in Node:**
   - `mat4.multiply(mat4.identity(), M)` returns `M` for all three demo mat4 sets.
   - `mat4.lookAt([0,0,5],[0,0,0],[0,1,0])` produces a matrix whose `m[14]` (translation z) is approximately `−5`.
   - All easing functions return `0` at `t=0` and `1` at `t=1`.
   - `cubicBezier(0.42,0,0.58,1)(0.5)` is close to `0.5` (symmetric curve).
   - Linear lerp: `lerp(a,b,0)===a`, `lerp(a,b,1)===b`.
3. **GLSL review:** `#version 300 es` first, no leading whitespace; `precision highp float;` in fragment shader; no GLSL ES 1.00 leftovers (`attribute`, `varying`, `gl_FragColor`, `texture2D`); every uniform used is declared and has a cached location.
4. **No external dependencies:** `grep` for `<script src`, `@import`, `fetch(` inside the script (only the logo `<img>` and `<a href>` may reference URLs).
5. **Template fidelity:** header/footer text matches 3.1 exactly (aside from demo name); `<title>` is `Computeranimation – <Demo name> Demo`.
6. **Checklist per file:** all control ids from the spec exist and are wired; no console errors expected on load; error banner path exists; resize handling present; `// ANIMATION CONCEPT:` markers present at each key animation idea; no exercises or tasks anywhere.

---

## 8. Delivery workflow

1. Read this file fully. Confirm the list of files received (Section 0).
2. Build **one demo at a time** in the order of Section 4. Write the file to `/mnt/user-data/outputs/<filename>`.
3. Run the quality gates (Section 7). Fix issues.
4. Present the file with `present_files`. Reply with **at most**: one sentence about what the demo shows, a short list of anything that could not be verified (e.g. no GPU in the sandbox), and a question whether to continue with the next demo.
5. Do not change file names, ids or layout without asking.
