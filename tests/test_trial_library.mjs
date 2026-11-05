import test from 'node:test';
import assert from 'node:assert/strict';
import {matchesTrial, normalizeSearch} from '../scripts/templates/trial_library.mjs';

const normal = {category:'normal', environment:'true', search:'Ntrial3 NTrial_3'};
const simulation = {category:'parkinson', environment:'true', search:'Fuchs22 Fuchs22'};
const scan = {category:'reconstruction', environment:'true', search:'RoomScan3c 3c6ed915-3279-4edf'};
const archive = {category:'archive', environment:'false', search:'trial4 32downstairs - speed'};

test('search tolerates case, separators and multiple terms', () => {
  for (const query of ['ntrial3', 'N TRIAL 3', 'NTrial_3', 'ntrial 3']) assert.ok(matchesTrial(normal, {query}));
  assert.ok(matchesTrial(scan, {query:'3c6ed915'}));
  assert.ok(matchesTrial(archive, {query:'downstairs speed'}));
  assert.ok(!matchesTrial(normal, {query:'Ntrial99'}));
  assert.equal(normalizeSearch('Ｆｕｃｈｓ２２'), 'fuchs22');
});

test('purpose and 3D filter are independent and combine with search', () => {
  const records = [normal, simulation, scan, archive];
  assert.equal(records.filter(t => matchesTrial(t, {environment:true})).length, 3);
  assert.deepEqual(records.filter(t => matchesTrial(t, {group:'reconstruction'})), [scan]);
  assert.deepEqual(records.filter(t => matchesTrial(t, {group:'parkinson',environment:true,query:'fuchs'})), [simulation]);
  assert.deepEqual(records.filter(t => matchesTrial(t, {group:'normal',query:'fuchs'})), []);
});

test('clear filters restores all records, including records without a dashboard', () => {
  for (const record of [normal, simulation, scan, archive]) assert.ok(matchesTrial(record));
  assert.ok(matchesTrial(archive, {group:'archive',query:'trial4'}));
  assert.ok(!matchesTrial(archive, {environment:true}));
});
