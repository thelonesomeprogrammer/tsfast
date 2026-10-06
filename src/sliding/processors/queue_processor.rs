use crate::common::ColumnState;
use crate::types::Compute;

pub struct QueueProcessor;

impl QueueProcessor {
    #[inline(always)]
    pub fn reset_state(state: &mut ColumnState, window_size: usize) {
        state.min_queue = vec![(0, 0.0); window_size];
        state.max_queue = vec![(0, 0.0); window_size];
        state.min_q_head = 0;
        state.min_q_tail = 0;
        state.min_q_len = 0;
        state.max_q_head = 0;
        state.max_q_tail = 0;
        state.max_q_len = 0;
    }

    #[inline(always)]
    pub fn update_batch(
        compute: Compute,
        new_slice: &[f32],
        global_start_idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        if compute.intersects(Compute::MIN | Compute::MAX | Compute::IQR) {
            for (i, &val) in new_slice.iter().enumerate() {
                let idx = global_start_idx + i;

                // Remove elements out of window
                if state.min_q_len > 0
                    && idx >= window_size
                    && state.min_queue[state.min_q_head].0 <= idx - window_size
                {
                    state.min_q_head += 1;
                    if state.min_q_head >= window_size {
                        state.min_q_head -= window_size;
                    }
                    state.min_q_len -= 1;
                }
                if state.max_q_len > 0
                    && idx >= window_size
                    && state.max_queue[state.max_q_head].0 <= idx - window_size
                {
                    state.max_q_head += 1;
                    if state.max_q_head >= window_size {
                        state.max_q_head -= window_size;
                    }
                    state.max_q_len -= 1;
                }

                // Min Queue
                while state.min_q_len > 0 {
                    let prev_tail = if state.min_q_tail == 0 {
                        window_size - 1
                    } else {
                        state.min_q_tail - 1
                    };
                    if state.min_queue[prev_tail].1 >= val {
                        state.min_q_tail = prev_tail;
                        state.min_q_len -= 1;
                    } else {
                        break;
                    }
                }
                state.min_queue[state.min_q_tail] = (idx, val);
                state.min_q_tail += 1;
                if state.min_q_tail >= window_size {
                    state.min_q_tail -= window_size;
                }
                state.min_q_len += 1;

                // Max Queue
                while state.max_q_len > 0 {
                    let prev_tail = if state.max_q_tail == 0 {
                        window_size - 1
                    } else {
                        state.max_q_tail - 1
                    };
                    if state.max_queue[prev_tail].1 <= val {
                        state.max_q_tail = prev_tail;
                        state.max_q_len -= 1;
                    } else {
                        break;
                    }
                }
                state.max_queue[state.max_q_tail] = (idx, val);
                state.max_q_tail += 1;
                if state.max_q_tail >= window_size {
                    state.max_q_tail -= window_size;
                }
                state.max_q_len += 1;
            }
            state.min_value = state.min_queue[state.min_q_head].1;
            state.max_value = state.max_queue[state.max_q_head].1;
        }
    }

    #[inline(always)]
    pub fn update_incremental(
        compute: Compute,
        new_val: f32,
        global_idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        if compute.intersects(Compute::MIN | Compute::MAX | Compute::IQR) {
            if state.min_queue.is_empty() {
                state.min_queue = vec![(0, 0.0); window_size];
                state.max_queue = vec![(0, 0.0); window_size];
            }

            // Remove elements out of window
            if state.min_q_len > 0
                && global_idx >= window_size
                && state.min_queue[state.min_q_head].0 <= global_idx - window_size
            {
                state.min_q_head += 1;
                if state.min_q_head >= window_size {
                    state.min_q_head -= window_size;
                }
                state.min_q_len -= 1;
            }
            if state.max_q_len > 0
                && global_idx >= window_size
                && state.max_queue[state.max_q_head].0 <= global_idx - window_size
            {
                state.max_q_head += 1;
                if state.max_q_head >= window_size {
                    state.max_q_head -= window_size;
                }
                state.max_q_len -= 1;
            }

            // Min Queue: remove larger elements from tail
            while state.min_q_len > 0 {
                let prev_tail = if state.min_q_tail == 0 {
                    window_size - 1
                } else {
                    state.min_q_tail - 1
                };
                if state.min_queue[prev_tail].1 >= new_val {
                    state.min_q_tail = prev_tail;
                    state.min_q_len -= 1;
                } else {
                    break;
                }
            }
            state.min_queue[state.min_q_tail] = (global_idx, new_val);
            state.min_q_tail += 1;
            if state.min_q_tail >= window_size {
                state.min_q_tail -= window_size;
            }
            state.min_q_len += 1;

            // Max Queue: remove smaller elements from tail
            while state.max_q_len > 0 {
                let prev_tail = if state.max_q_tail == 0 {
                    window_size - 1
                } else {
                    state.max_q_tail - 1
                };
                if state.max_queue[prev_tail].1 <= new_val {
                    state.max_q_tail = prev_tail;
                    state.max_q_len -= 1;
                } else {
                    break;
                }
            }
            state.max_queue[state.max_q_tail] = (global_idx, new_val);
            state.max_q_tail += 1;
            if state.max_q_tail >= window_size {
                state.max_q_tail -= window_size;
            }
            state.max_q_len += 1;

            state.min_value = state.min_queue[state.min_q_head].1;
            state.max_value = state.max_queue[state.max_q_head].1;
        }
    }

    #[inline(always)]
    pub fn process_remainder(
        compute: Compute,
        val: f32,
        idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        if compute.intersects(Compute::MIN | Compute::MAX | Compute::IQR) {
            // Min Queue
            while state.min_q_len > 0 {
                let prev_tail = if state.min_q_tail == 0 {
                    window_size - 1
                } else {
                    state.min_q_tail - 1
                };
                if state.min_queue[prev_tail].1 >= val {
                    state.min_q_tail = prev_tail;
                    state.min_q_len -= 1;
                } else {
                    break;
                }
            }
            state.min_queue[state.min_q_tail] = (idx, val);
            state.min_q_tail += 1;
            if state.min_q_tail >= window_size {
                state.min_q_tail -= window_size;
            }
            state.min_q_len += 1;

            // Max Queue
            while state.max_q_len > 0 {
                let prev_tail = if state.max_q_tail == 0 {
                    window_size - 1
                } else {
                    state.max_q_tail - 1
                };
                if state.max_queue[prev_tail].1 <= val {
                    state.max_q_tail = prev_tail;
                    state.max_q_len -= 1;
                } else {
                    break;
                }
            }
            state.max_queue[state.max_q_tail] = (idx, val);
            state.max_q_tail += 1;
            if state.max_q_tail >= window_size {
                state.max_q_tail -= window_size;
            }
            state.max_q_len += 1;
        }
    }
}
