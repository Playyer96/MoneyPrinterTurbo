Added `get_active_task_count()` to state.py:
- Abstract method added to BaseState
- MemoryState implementation: iterates _tasks dict under lock, counts those with state == TASK_STATE_PROCESSING
- RedisState implementation: scans all HASH keys, checks "state" field, counts matches
