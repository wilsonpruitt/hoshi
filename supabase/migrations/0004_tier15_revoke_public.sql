-- HOSHI — Tier 1.5 lock-down, corrected.
--
-- 0003 revoked submit_move/report_result from anon + authenticated, but Postgres
-- grants function EXECUTE to PUBLIC by default and anon inherits through PUBLIC,
-- so those calls were still reachable. Revoke from PUBLIC to actually close them.
-- Both are dead code now: the submit-move Edge Function writes moves and sets the
-- winner directly with the service-role key, so nothing legitimate calls them.

revoke execute on function public.submit_move(uuid,uuid,jsonb) from public;
revoke execute on function public.report_result(uuid,uuid,int) from public;
