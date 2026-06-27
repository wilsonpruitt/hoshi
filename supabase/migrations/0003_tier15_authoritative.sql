-- HOSHI — Tier 1.5: make the submit-move Edge Function the sole move-writer.
--
-- APPLY THIS LAST: only after the submit-move function is deployed AND the client
-- has been cut over to call it. Until then the old client still uses submit_move,
-- and revoking it early would break live games.
--
-- The Edge Function uses the service-role key (bypasses RLS) to append moves and
-- to set the winner authoritatively. Revoking these from anon/authenticated closes
-- the only paths a hand-crafted client could use to write an UNVALIDATED move or
-- forge a result — legality and scoring now live entirely in the function.

revoke execute on function public.submit_move(uuid,uuid,jsonb)  from anon, authenticated;
revoke execute on function public.report_result(uuid,uuid,int)  from anon, authenticated;
