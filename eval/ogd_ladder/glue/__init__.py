"""One module per competition bot: ``Glue().handle(msg) -> actIndex | None``.

Each glue stands in for the team's websocket client, which the published bot
sources do not include: it feeds every server message to the bot's own
``State`` and asks its ``Action`` for an index when the message carries an
``actionList``, the way the competition's example client did. Glue modules are
imported by ``bot_host`` with the bot's private working copy at the head of
``sys.path``, so ``import state`` / ``import action`` resolve to that bot.
"""
