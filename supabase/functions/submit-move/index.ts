// Hoshi — authoritative move submission (Tier 1.5).
//
// The sole move-writer. Loads the game's move log, replays it through the SAME
// rules engine the browser uses, and rejects anything illegal or out-of-turn
// BEFORE appending. When a move ends the game it writes the winner directly, so
// results no longer depend on both clients agreeing. Uses the service-role key
// (auto-injected) to bypass RLS; the seat_token is the capability that authorizes
// a move, so this function is deployed public (verify_jwt = false).
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { Hoshi } from "../_shared/engine.js";

const cors = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { ...cors, "Content-Type": "application/json" } });
// Envelope: validation outcomes are 200 with {ok}, so the client reads data.ok
// instead of unwrapping HTTP errors. Only true crashes are non-2xx.
const fail = (error: string) => json({ ok: false, error });
const done = (obj: Record<string, unknown>) => json({ ok: true, ...obj });

// stored initial_state has history as an array (client stateToObj); engine wants a Set
const fromObj = (o: any) => ({ ...o, history: new Set(o.history || []) });

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors });
  try {
    const { game_id, seat_token, move } = await req.json();
    if (!game_id || !seat_token || !move || !["trooper", "drone", "pass"].includes(move.type))
      return fail("bad request");

    const supa = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);

    const { data: game, error: gErr } = await supa
      .from("games").select("initial_state,status").eq("id", game_id).single();
    if (gErr || !game) return fail("no such game");
    if (game.status !== "active") return fail("game is not active");

    const { data: seatRow } = await supa
      .from("game_seats").select("player").eq("game_id", game_id).eq("token", seat_token).maybeSingle();
    if (!seatRow) return fail("bad seat token");
    const seat = seatRow.player;

    const { data: rows, error: mErr } = await supa
      .from("moves").select("ply,move").eq("game_id", game_id).order("ply");
    if (mErr) return json({ error: "load failed" }, 500);

    // replay the authoritative log
    let s: any = fromObj(game.initial_state);
    for (const r of rows!) { const ns = Hoshi.apply(s, r.move); if (ns) s = ns; }

    // authoritative checks: turn, move-shape, legality
    if (seat !== s.toMove) return fail("not your turn");
    if (move.type !== "pass" && (!Number.isInteger(move.i) || move.i < 0 || move.i >= s.N * s.N))
      return fail("bad move index");
    const ns = Hoshi.apply(s, move);
    if (ns === null) return fail("illegal move");

    // append (unique(game_id,ply) makes concurrent submissions safe)
    const ply = rows!.length;
    const { error: insErr } = await supa.from("moves").insert({ game_id, ply, player: seat, move });
    if (insErr) return fail("move already taken");

    if (ns.over) {
      await supa.from("games").update({ status: "over", winner: ns.winner }).eq("id", game_id);
    }
    return done({ ply, over: !!ns.over, winner: ns.over ? ns.winner : null });
  } catch (e) {
    return json({ error: String(e) }, 500);
  }
});
