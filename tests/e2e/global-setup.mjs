// Builds the synthetic Fuchs-shaped trials before the browser tests start (see prepare_fixtures.py).
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export default function globalSetup() {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const python = process.env.PD_INSIGHTER_PYTHON || path.resolve(here, '..', '..', '..', '.venv', 'Scripts', 'python.exe');
  execFileSync(python, [path.join(here, 'prepare_fixtures.py')], {stdio: 'inherit', env: {...process.env, PYTHONUTF8: '1'}});
}
