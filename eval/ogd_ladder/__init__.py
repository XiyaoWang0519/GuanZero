"""Ladder against the 1st National Guandan AI competition bots, on OpenGuanDan.

The OpenGuanDan table (``eval.ogd_adapter.bridge``) referees: it is the NJUPT
simulator whose message format the competition bots were written against.
Competition bots run in their own subprocess (``bot_host``) behind a small
per-bot glue module (``glue/``) that replaces each team's missing websocket
client. Our policy plays through a ``gd`` mirror of every round
(``ours.OurSeats``), exactly as ``eval.danlm.arena`` does for DanLM.

Nothing from the bot collection is vendored: ``BOTS_ROOT`` points at the
``baselines/`` directory of a DanLM checkout (github.com/dashidhy/DanLM).
"""
