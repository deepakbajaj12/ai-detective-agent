import { apiFetch } from './apiBase';

export async function startIndexRefresh(caseId = 'default') {
  const res = await apiFetch('/api/jobs/index_refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ case_id: caseId }),
  });
  return res.job_id;
}

export async function startTransformerTrain(trainingJson = 'inputs/sample_training.json') {
  const res = await apiFetch('/api/jobs/transformer_train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ training_json: trainingJson }),
  });
  return res.job_id;
}

export async function getJobStatus(jobId) {
  return await apiFetch(`/api/jobs/${jobId}`);
}

export async function listJobs(limit = 50) {
  return await apiFetch(`/api/jobs?limit=${limit}`);
}

export async function cancelJob(jobId) {
  return await apiFetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
}
