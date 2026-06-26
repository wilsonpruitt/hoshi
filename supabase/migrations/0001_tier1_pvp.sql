-- HOSHI — Tier 1 PvP: server-held move log, capability-URL seats, no accounts.
--
-- Model: the source of truth is `games.initial_state` (the seeded opening, which
-- pins the random `fog` variant) plus an append-only `moves` log. Position is
-- never stored — clients REPLAY initial_state + moves through hoshi-rules.js.
--
-- Authority (Tier 1 trust model): Postgres can enforce *turn order* (Blue=0 first,
-- every non-pass move flips to-move, so next = move_count % 2) but CANNOT run the
-- JS rules engine, so it does NOT check move legality. Each client revalidates by
-- replaying the log. Move legality goes server-authoritative in Tier 1.5 via a
-- Supabase Edge Function that imports hoshi-rules.js (the engine is already
-- DOM-free), validating before the insert. Until then: casual / friends-with-links.
--
-- Secrets: each game mints two seat tokens (uuid capability URLs). They live in
-- `game_seats`, which is locked to RLS-deny — ONLY the SECURITY DEFINER RPCs read
-- them. `games` and `moves` are world-readable by game id (free spectating); the
-- only thing a token gates is the right to MOVE for a seat.

create extension if not exists pgcrypto;   -- gen_random_uuid() (Supabase ships this)

-- ────────────────────────────────────────────────────────────────────────────
-- Tables
-- ────────────────────────────────────────────────────────────────────────────

create table if not exists public.games (
  id            uuid primary key default gen_random_uuid(),
  initial_state jsonb       not null,           -- the seeded opening State (replay root)
  board_n       int         not null,           -- metadata for display/listing
  variant0      text        not null,
  variant1      text        not null,
  status        text        not null default 'active'   -- 'active' | 'over'
                  check (status in ('active','over')),
  winner        int         check (winner in (-1,0,1)), -- null until reported; -1 = draw
  created_at    timestamptz not null default now()
);

-- Seat capabilities. NOT readable by clients — RLS denies all; only the
-- SECURITY DEFINER RPCs (which bypass RLS) ever touch this table.
create table if not exists public.game_seats (
  game_id         uuid not null references public.games(id) on delete cascade,
  player          int  not null check (player in (0,1)),
  token           uuid not null default gen_random_uuid(),
  reported_winner int  check (reported_winner in (-1,0,1)),  -- this seat's replayed result
  primary key (game_id, player)
);
create unique index if not exists game_seats_token_idx on public.game_seats(token);

-- Append-only move log. `ply` is just the sequence number (= prior move_count);
-- the engine recomputes its own internal ply on replay. The unique(game_id,ply)
-- constraint is the race guard: two simultaneous inserts for the same turn — one
-- wins, the other gets a duplicate-key error.
create table if not exists public.moves (
  game_id    uuid        not null references public.games(id) on delete cascade,
  ply        int         not null,
  player     int         not null check (player in (0,1)),
  move       jsonb       not null,            -- {type:'trooper'|'drone'|'pass', i?:int}
  created_at timestamptz not null default now(),
  primary key (game_id, ply)
);
create index if not exists moves_game_idx on public.moves(game_id, ply);

-- ────────────────────────────────────────────────────────────────────────────
-- Row-level security
-- ────────────────────────────────────────────────────────────────────────────

alter table public.games      enable row level security;
alter table public.moves      enable row level security;
alter table public.game_seats enable row level security;

-- games / moves: anyone can READ by id (spectating). No direct writes — all
-- mutations go through the RPCs below (SECURITY DEFINER bypasses these policies).
create policy games_read on public.games for select using (true);
create policy moves_read on public.moves for select using (true);
-- game_seats: NO policies → with RLS enabled, every client read/write is denied.
-- (Intentional: only the definer RPCs may resolve a token to a seat.)

-- ────────────────────────────────────────────────────────────────────────────
-- RPCs (the only write path)
-- ────────────────────────────────────────────────────────────────────────────

-- Create a game. Client builds the seeded opening with hoshi-rules.js newState()
-- (which fixes the fog RNG to a concrete board) and passes it in. Returns both
-- seat tokens once; creator keeps blue, sends the red link to the opponent.
create or replace function public.create_game(
  p_initial_state jsonb,
  p_board_n       int,
  p_variant0      text,
  p_variant1      text
) returns table (game_id uuid, blue_token uuid, red_token uuid)
language plpgsql security definer set search_path = public as $$
declare v_id uuid; v_blue uuid; v_red uuid;
begin
  insert into games(initial_state, board_n, variant0, variant1)
    values (p_initial_state, p_board_n, p_variant0, p_variant1)
    returning id into v_id;
  insert into game_seats(game_id, player) values (v_id, 0) returning token into v_blue;
  insert into game_seats(game_id, player) values (v_id, 1) returning token into v_red;
  return query select v_id, v_blue, v_red;
end $$;

-- Submit a move. Enforces: game active, token owns a seat, it's that seat's turn,
-- ply is next. Does NOT check legality (client revalidates). The unique constraint
-- backstops races. A 'pass' move ends the game (status→over); the winner is filled
-- in by report_result after the client scores the replay.
create or replace function public.submit_move(
  p_game_id    uuid,
  p_seat_token uuid,
  p_move       jsonb
) returns int                                    -- the ply that was written
language plpgsql security definer set search_path = public as $$
declare
  v_seat   int;
  v_start  int;
  v_count  int;
  v_next   int;
  v_status text;
begin
  select status, (initial_state->>'toMove')::int
    into v_status, v_start
    from games where id = p_game_id;
  if not found then raise exception 'no such game'; end if;
  if v_status <> 'active' then raise exception 'game is over'; end if;

  select player into v_seat
    from game_seats where game_id = p_game_id and token = p_seat_token;
  if not found then raise exception 'bad seat token'; end if;

  select count(*) into v_count from moves where game_id = p_game_id;
  v_next := (v_start + v_count) % 2;
  if v_seat <> v_next then raise exception 'not your turn'; end if;

  insert into moves(game_id, ply, player, move)
    values (p_game_id, v_count, v_seat, p_move);   -- racing insert → duplicate-key error

  if p_move->>'type' = 'pass' then
    update games set status = 'over' where id = p_game_id;
  end if;
  return v_count;
end $$;

-- Record the result after a client detects game-over by replay (pass-score,
-- decapitation, or maxPlies). A single seat CANNOT set the winner — it only
-- records its own replayed result; the game finalizes only when BOTH seats
-- report the same value. Both clients replay the identical deterministic log, so
-- the honest result always agrees; a client reporting a self-serving winner just
-- fails to match its opponent and nothing is committed (a sore loser can stall
-- finalization, but cannot forge a result — that nuisance closes in Tier 1.5
-- when scoring moves server-side). Idempotent.
create or replace function public.report_result(
  p_game_id    uuid,
  p_seat_token uuid,
  p_winner     int
) returns void
language plpgsql security definer set search_path = public as $$
declare v_seat int; v_other int;
begin
  if p_winner not in (-1,0,1) then raise exception 'bad winner'; end if;
  select player into v_seat
    from game_seats where game_id = p_game_id and token = p_seat_token;
  if not found then raise exception 'bad seat token'; end if;

  update game_seats set reported_winner = p_winner
    where game_id = p_game_id and player = v_seat;

  select reported_winner into v_other
    from game_seats where game_id = p_game_id and player = 1 - v_seat;
  if v_other is not null and v_other = p_winner then        -- both seats agree
    update games set status = 'over', winner = p_winner
      where id = p_game_id and winner is null;
  end if;
end $$;

-- Anonymous clients (Supabase anon key) may call the RPCs but cannot touch tables
-- directly except the read policies above.
grant execute on function public.create_game(jsonb,int,text,text)  to anon, authenticated;
grant execute on function public.submit_move(uuid,uuid,jsonb)      to anon, authenticated;
grant execute on function public.report_result(uuid,uuid,int)      to anon, authenticated;

-- ────────────────────────────────────────────────────────────────────────────
-- Realtime: push move inserts + status flips to subscribed clients
-- ────────────────────────────────────────────────────────────────────────────
alter publication supabase_realtime add table public.moves;
alter publication supabase_realtime add table public.games;
