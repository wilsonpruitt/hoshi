-- HOSHI — Tier 2 lobby: publicly listed "open" games anyone can join.
--
-- Builds on 0001. A game can now start as status='open' (host waiting). The open
-- list is just the world-readable games table filtered to status='open' — seat
-- tokens stay sealed in game_seats and are handed out ONLY by join_open_game,
-- atomically (first caller wins). Nothing here exposes a token to the list.

-- ── games.status gains 'open'; add an optional host display name ─────────────
alter table public.games drop constraint if exists games_status_check;
alter table public.games add  constraint games_status_check
  check (status in ('open','active','over'));
alter table public.games add column if not exists host_name text;

-- ── Create a LISTED game. Host keeps Blue; the Red token is withheld (claimed
--    on join). Returns only the host's token. ────────────────────────────────
create or replace function public.create_open_game(
  p_initial_state jsonb,
  p_board_n       int,
  p_variant0      text,
  p_variant1      text,
  p_host_name     text default null
) returns table (game_id uuid, blue_token uuid)
language plpgsql security definer set search_path = public as $$
declare v_id uuid; v_blue uuid;
begin
  insert into games(initial_state, board_n, variant0, variant1, status, host_name)
    values (p_initial_state, p_board_n, p_variant0, p_variant1, 'open', p_host_name)
    returning id into v_id;
  insert into game_seats(game_id, player) values (v_id, 0) returning token into v_blue;
  insert into game_seats(game_id, player) values (v_id, 1);   -- Red seat; token claimed on join
  return query select v_id, v_blue;
end $$;

-- ── Join a listed game. Atomic: only the first caller flips open→active and
--    receives the Red token; everyone else gets 'game no longer open'. (The
--    UPDATE ... WHERE status='open' row-locks, so concurrent joins can't both
--    win.) submit_move already blocks moves while status<>'active', so Blue
--    can't play into an empty lobby before someone joins. ────────────────────
create or replace function public.join_open_game(p_game_id uuid)
returns uuid                                   -- the Red seat token
language plpgsql security definer set search_path = public as $$
declare v_rows int; v_token uuid;
begin
  update games set status='active' where id = p_game_id and status='open';
  get diagnostics v_rows = row_count;
  if v_rows = 0 then raise exception 'game no longer open'; end if;
  select token into v_token from game_seats where game_id = p_game_id and player = 1;
  return v_token;
end $$;

-- ── Host withdraws an unclaimed game (only while still open). ────────────────
create or replace function public.cancel_open_game(
  p_game_id    uuid,
  p_blue_token uuid
) returns void
language plpgsql security definer set search_path = public as $$
begin
  if not exists (select 1 from game_seats
                 where game_id = p_game_id and player = 0 and token = p_blue_token)
    then raise exception 'bad host token'; end if;
  update games set status='over' where id = p_game_id and status='open';
end $$;

grant execute on function public.create_open_game(jsonb,int,text,text,text) to anon, authenticated;
grant execute on function public.join_open_game(uuid)                       to anon, authenticated;
grant execute on function public.cancel_open_game(uuid,uuid)               to anon, authenticated;
