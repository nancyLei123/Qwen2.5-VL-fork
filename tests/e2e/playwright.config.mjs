import {defineConfig} from '@playwright/test';
import {mkdtempSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

// Starts this checkout's server on its own port with trial data from runs/ (Parki is required, see
// docs/symptom_annotation/README.md). Symptom labels and annotations go to a fresh temporary
// folder, so tests never touch real annotations. Override the interpreter with PD_INSIGHTER_PYTHON.
const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const python = process.env.PD_INSIGHTER_PYTHON || path.resolve(repo, '..', '.venv', 'Scripts', 'python.exe');
const store = process.env.PD_INSIGHTER_E2E_STORE || mkdtempSync(path.join(os.tmpdir(), 'pd-symptom-e2e-'));
process.env.PD_INSIGHTER_E2E_STORE = store;
process.env.PD_INSIGHTER_E2E_RUNS = path.join(repo, 'runs');
const port = Number(process.env.PD_INSIGHTER_E2E_PORT || 8792);

export default defineConfig({
  testDir: here,
  testMatch: /.*\.spec\.mjs/,
  globalSetup: path.join(here, 'global-setup.mjs'),
  timeout: 90_000,
  expect: {timeout: 15_000},
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${port}/Qwen2.5-VL-fork/runs/`,
    viewport: {width: 1440, height: 1000},
    launchOptions: {args: ['--autoplay-policy=no-user-gesture-required']},
  },
  webServer: {
    command: `"${python}" "${path.join(repo, 'scripts', 'pd_insighter_server.py')}" --host 127.0.0.1 --port ${port} --symptom-store "${store}"`,
    url: `http://127.0.0.1:${port}/api/symptom-labels`,
    env: {PYTHONUTF8: '1'},
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
