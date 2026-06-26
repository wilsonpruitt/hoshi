// Hoshi online (Tier 1) — Supabase connection. COMMITTED ON PURPOSE.
// The anon / "publishable" key belongs in the browser: data access is gated by
// Row-Level Security (supabase/migrations/0001_tier1_pvp.sql), not by hiding the
// key. (The secret service_role key is never used client-side and is not here.)
// Get the anon key from the Supabase dashboard:
//   Project Settings → API → Project API keys → `anon` `public`.
window.HOSHI_SUPABASE = {
  url: "https://hzwmansavmaqvtfnpllh.supabase.co",
  anonKey: "sb_publishable_suqgLQUA5Opa0QIfyNa_kg_STYkqnHY",   // ← paste it here, then we commit
};
