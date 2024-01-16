# LLM Fact Log

The experiment here was to convert from a natural language statement into a list of facts, then insert them into an append only log. Then to convert a natural language query into a series of facts to assert.

The current implementation doesn't work very well. I believe that it may be better with an extra step that checks if queries should be merged & takes into account the full statement / query rather than just the individual fields.

There could also be a post step that asks if the question is true given the data.
