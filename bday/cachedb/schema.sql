-- cachedb/schema.sql
PRAGMA journal_mode=WAL;

-- 1) LLM cache ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_cache (
  id INTEGER PRIMARY KEY,
  model TEXT,
  prompt_norm_hash TEXT,
  prompt_text TEXT,
  tool_state TEXT,
  output_text TEXT,
  usage_json TEXT,
  ts REAL,
  ttl INTEGER,
  source_tag TEXT,
  version TEXT
);

CREATE TABLE IF NOT EXISTS llm_cache_vectors (
  id INTEGER PRIMARY KEY,
  llm_cache_id INTEGER,
  prompt_vec_json TEXT,
  FOREIGN KEY(llm_cache_id) REFERENCES llm_cache(id) ON DELETE CASCADE
);

-- 2) Answers -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS answers (
  id INTEGER PRIMARY KEY,
  canonical_q TEXT,
  question_text TEXT,
  answer_text TEXT,
  confidence REAL,
  freshness_horizon INTEGER,
  ts REAL,
  evidence_json TEXT,
  sources_json TEXT
);

CREATE TABLE IF NOT EXISTS answers_vectors (
  id INTEGER PRIMARY KEY,
  answer_id INTEGER,
  q_vec_json TEXT,
  FOREIGN KEY(answer_id) REFERENCES answers(id) ON DELETE CASCADE
);

-- 3) Plans -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS plans (
  id INTEGER PRIMARY KEY,
  intent_key TEXT,
  goal_text TEXT,
  site_domain TEXT,
  env_fingerprint TEXT,
  plan_json TEXT,
  success_rate REAL,
  version TEXT,
  ts REAL
);

CREATE TABLE IF NOT EXISTS plans_vectors (
  id INTEGER PRIMARY KEY,
  plan_id INTEGER,
  goal_vec_json TEXT,
  FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE
);

-- 4) DOM chunks --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dom_chunks (
  id INTEGER PRIMARY KEY,
  site_domain TEXT,
  selector TEXT,
  role TEXT,
  text TEXT,
  attrs_json TEXT,
  ts REAL,
  quality_score REAL
);

CREATE TABLE IF NOT EXISTS dom_vectors (
  id INTEGER PRIMARY KEY,
  dom_id INTEGER,
  dom_vec_json TEXT,
  FOREIGN KEY(dom_id) REFERENCES dom_chunks(id) ON DELETE CASCADE
);
