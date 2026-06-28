-- HOSHI — analytics: one snapshot row per FINISHED game, all modes.
--
-- Distinct from the Tier-1 online tables (games/moves), which only capture live
-- PvP. This logs every completed game — vs-CPU, hot-seat, watch (cpu-vs-cpu),
-- by-link, and online — so openings and end-states can be analyzed later (and,
-- in particular, used to weight the bot's opening book by REAL opening frequency).
--
-- Privacy/abuse posture matches the rest of the schema: the table is RLS-locked
-- with NO direct policies, so anon clients can't read or write it. The only write
-- path is the SECURITY DEFINER log_game() RPC (insert-only). No PII is recorded —
-- just config, result, and the move list. A per-browser random client_id allows
-- coarse de-duplication/segmentation without identifying anyone.

create table if not exists public.game_log (
  id             uuid        primary key default gen_random_uuid(),
  created_at     timestamptz not null default now(),
  source         text,                          -- vscpu | hotseat | cpucpu | online | bylink
  board_n        int,
  drone_range    int,
  variant0       text,                          -- setup variant per seat (local play: both equal)
  variant1       text,
  encircle       boolean,                       -- rule dials (default off)
  mobile         boolean,
  blue_player    text,                          -- human | cpu
  red_player     text,
  blue_tier      text,                          -- difficulty/style when that seat is cpu, else null
  red_tier       text,
  winner         int,                           -- -1 draw | 0 blue | 1 red
  win_type       text,                          -- decap | territory | maxplies
  score_blue     int,                           -- area score at game end
  score_red      int,
  plies          int,
  moves          jsonb,                         -- ordered move list: [{type,i} | {type:'pass'} | {type:'move',from,to}]
  client_id      text,                          -- random per-browser id (analytics only, not PII)
  online_game_id uuid,                          -- set for online games; used to de-dupe the two seats' logs
  app_commit     text
);

-- Online games are logged by BOTH seats — de-dupe on the shared game id. Local
-- games carry a null id; Postgres allows many null rows under a partial unique
-- index, so they're never collapsed.
create unique index if not exists game_log_online_uniq
  on public.game_log(online_game_id) where online_game_id is not null;
create index if not exists game_log_created_idx on public.game_log(created_at);

alter table public.game_log enable row level security;
-- No policies → with RLS on, every direct client read/write is denied. The RPC
-- below (SECURITY DEFINER) is the sole write path.

create or replace function public.log_game(p jsonb)
returns void
language plpgsql security definer set search_path = public as $$
begin
  insert into public.game_log(
    source, board_n, drone_range, variant0, variant1, encircle, mobile,
    blue_player, red_player, blue_tier, red_tier, winner, win_type,
    score_blue, score_red, plies, moves, client_id, online_game_id, app_commit)
  values (
    p->>'source',
    nullif(p->>'board_n','')::int,
    nullif(p->>'drone_range','')::int,
    p->>'variant0', p->>'variant1',
    nullif(p->>'encircle','')::boolean, nullif(p->>'mobile','')::boolean,
    p->>'blue_player', p->>'red_player', p->>'blue_tier', p->>'red_tier',
    nullif(p->>'winner','')::int, p->>'win_type',
    nullif(p->>'score_blue','')::int, nullif(p->>'score_red','')::int,
    nullif(p->>'plies','')::int,
    p->'moves',
    p->>'client_id',
    nullif(p->>'online_game_id','')::uuid,
    p->>'app_commit')
  on conflict (online_game_id) where online_game_id is not null do nothing;
end $$;

grant execute on function public.log_game(jsonb) to anon, authenticated;
