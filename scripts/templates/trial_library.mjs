export function normalizeSearch(value) {
  return String(value).normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

export function matchesTrial(trial, {query = '', group = 'all', environment = false} = {}) {
  if (group !== 'all' && trial.category !== group) return false;
  if (environment && trial.environment !== 'true') return false;
  const text = normalizeSearch(trial.search);
  const compact = text.replaceAll(' ', '');
  return normalizeSearch(query).split(' ').filter(Boolean).every(word => text.includes(word) || compact.includes(word));
}

if (typeof document !== 'undefined') {
  const search = document.getElementById('trialSearch');
  const environment = document.getElementById('hasEnvironment');
  const cards = [...document.querySelectorAll('.trial-card')];
  const sections = [...document.querySelectorAll('[data-section]')];
  const groups = [...document.querySelectorAll('[data-group]')];
  const clear = document.getElementById('clearFilters');
  const status = document.getElementById('resultsStatus');
  const names = {all:'all categories',normal:'Normal motion',parkinson:'Parkinson simulation',reconstruction:'Dedicated 3D scans',archive:'Earlier / unclassified'};
  let group = 'all';

  function applyFilters(save = true) {
    const filters = {query:search.value,group,environment:environment.checked};
    let count = 0;
    for (const card of cards) {
      card.hidden = !matchesTrial(card.dataset, filters);
      if (!card.hidden) count++;
    }
    for (const section of sections) {
      const visible = [...section.querySelectorAll('.trial-card')].filter(card => !card.hidden).length;
      section.hidden = visible === 0;
      section.querySelector('.section-count').textContent = visible;
    }
    for (const button of groups) button.setAttribute('aria-pressed', String(button.dataset.group === group));
    const active = Boolean(search.value || group !== 'all' || environment.checked);
    clear.hidden = !active;
    document.getElementById('emptyState').hidden = count !== 0;
    status.textContent = `${count} of ${cards.length} recordings / ${names[group]}${environment.checked ? ' / with 3D' : ''}`;
    if (save) {
      const url = new URL(location.href);
      for (const [key, value] of [['q',search.value.trim()],['group',group === 'all' ? '' : group],['3d',environment.checked ? '1' : '']]) {
        if (value) url.searchParams.set(key, value); else url.searchParams.delete(key);
      }
      history.replaceState(null, '', url);
    }
  }

  function restoreFilters() {
    const params = new URLSearchParams(location.search);
    search.value = params.get('q') || '';
    group = Object.hasOwn(names, params.get('group')) ? params.get('group') : 'all';
    environment.checked = params.get('3d') === '1';
    applyFilters(false);
  }
  search.addEventListener('input', () => applyFilters());
  environment.addEventListener('change', () => applyFilters());
  for (const button of groups) button.addEventListener('click', () => { group = button.dataset.group; applyFilters(); });
  clear.addEventListener('click', () => { search.value = ''; group = 'all'; environment.checked = false; applyFilters(); search.focus(); });
  window.addEventListener('popstate', restoreFilters);
  restoreFilters();
}
