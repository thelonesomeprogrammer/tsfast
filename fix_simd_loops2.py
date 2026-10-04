import re
with open('src/features/stationarity.rs', 'r') as f:
    content = f.read()

simd_loop2 = """                    for i in 0..k_vars {
                        for j in 0..y_len {
                            xty[i] += x_mat[[j, i]] * y_arr[j];
                        }
                        for j in i..k_vars {
                            let mut sum = 0.0;
                            for t in 0..y_len {
                                sum += x_mat[[t, i]] * x_mat[[t, j]];
                            }
                            xtx[[i, j]] = sum;
                            xtx[[j, i]] = sum;
                        }
                    }"""
# Wait, this might be in the second loop (re-evaluation with full data for best_lag)
new_simd_loop2 = """                    for i in 0..k_vars {
                        let mut sum_xty = 0.0;
                        let mut t = 0;
                        let mut sum_vec_xty = f64x8::splat(0.0);
                        while t + 8 <= y_len {
                            let x_vec = f64x8::from_array([
                                x_mat[[t, i]], x_mat[[t+1, i]], x_mat[[t+2, i]], x_mat[[t+3, i]],
                                x_mat[[t+4, i]], x_mat[[t+5, i]], x_mat[[t+6, i]], x_mat[[t+7, i]],
                            ]);
                            let y_vec = f64x8::from_slice(&y_arr.as_slice().unwrap()[t..t+8]);
                            sum_vec_xty += x_vec * y_vec;
                            t += 8;
                        }
                        sum_xty += sum_vec_xty.reduce_sum();
                        while t < y_len {
                            sum_xty += x_mat[[t, i]] * y_arr[t];
                            t += 1;
                        }
                        xty[i] = sum_xty;

                        for j in i..k_vars {
                            let mut sum = 0.0;
                            let mut t = 0;
                            let mut sum_vec = f64x8::splat(0.0);
                            while t + 8 <= y_len {
                                let xi_vec = f64x8::from_array([
                                    x_mat[[t, i]], x_mat[[t+1, i]], x_mat[[t+2, i]], x_mat[[t+3, i]],
                                    x_mat[[t+4, i]], x_mat[[t+5, i]], x_mat[[t+6, i]], x_mat[[t+7, i]],
                                ]);
                                let xj_vec = f64x8::from_array([
                                    x_mat[[t, j]], x_mat[[t+1, j]], x_mat[[t+2, j]], x_mat[[t+3, j]],
                                    x_mat[[t+4, j]], x_mat[[t+5, j]], x_mat[[t+6, j]], x_mat[[t+7, j]],
                                ]);
                                sum_vec += xi_vec * xj_vec;
                                t += 8;
                            }
                            sum += sum_vec.reduce_sum();
                            while t < y_len {
                                sum += x_mat[[t, i]] * x_mat[[t, j]];
                                t += 1;
                            }
                            xtx[[i, j]] = sum;
                            xtx[[j, i]] = sum;
                        }
                    }"""

content = content.replace(simd_loop2, new_simd_loop2)

with open('src/features/stationarity.rs', 'w') as f:
    f.write(content)
