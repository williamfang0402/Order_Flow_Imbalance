import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# Load and prepare the data
def load_and_prepare_data(filepath, timestamp_col='ts_event'):
    df = pd.read_csv(filepath)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


# Compute event-level OFI for a given level
def compute_event_level_ofi(df, level):
    bid_px = f'bid_px_0{level}'
    bid_sz = f'bid_sz_0{level}'
    ask_px = f'ask_px_0{level}'
    ask_sz = f'ask_sz_0{level}'

    of_b, of_a = [0], [0]
    for i in range(1, len(df)):
        bp_now, bp_prev = df.at[i, bid_px], df.at[i-1, bid_px]
        bs_now, bs_prev = df.at[i, bid_sz], df.at[i-1, bid_sz]
        ap_now, ap_prev = df.at[i, ask_px], df.at[i-1, ask_px]
        as_now, as_prev = df.at[i, ask_sz], df.at[i-1, ask_sz]

        if bp_now > bp_prev:
            ofb = bs_now
        elif bp_now == bp_prev:
            ofb = bs_now - bs_prev
        else:
            ofb = -bs_now

        if ap_now > ap_prev:
            ofa = -as_now
        elif ap_now == ap_prev:
            ofa = as_now - as_prev
        else:
            ofa = as_now

        of_b.append(ofb)
        of_a.append(ofa)

    return np.array(of_b) - np.array(of_a)


# Compute best-level OFI over time window (t-h, t]
def compute_best_level_ofi(df, h_seconds=1.0):
    df['ofi_event'] = compute_event_level_ofi(df, level=0)
    output = []
    for i, t in enumerate(df['ts_event']):
        t_start = t - pd.Timedelta(seconds=h_seconds)
        ofi = df.loc[(df['ts_event'] > t_start) & (df['ts_event'] <= t), 'ofi_event'].sum()
        output.append(ofi)
    df['ofi_best'] = output
    return df[['ts_event', 'ofi_best']]


# Compute multi-level OFI for levels 0–9
def compute_multi_level_ofi(df, levels=10):
    result = pd.DataFrame(index=df.index)
    for level in range(levels):
        result[f'ofi_l{level}'] = compute_event_level_ofi(df, level)
    return result


# Compute integrated OFI using PCA
def compute_integrated_ofi_via_pca(df_multi):
    df_norm = df_multi / (df_multi.abs().mean() + 1e-6)  # Normalize each level
    pca = PCA(n_components=1)
    pca.fit(df_norm)
    w1 = pca.components_[0]
    w1_l1 = np.sum(np.abs(w1))
    integrated_ofi = df_norm @ w1 / w1_l1
    return integrated_ofi


# Combine everything into one pipeline
def compute_all_ofi_features(filepath):
    df = load_and_prepare_data(filepath)
    df_best = compute_best_level_ofi(df.copy(), h_seconds=1.0)
    df_multi = compute_multi_level_ofi(df.copy(), levels=10)
    df_integrated = compute_integrated_ofi_via_pca(df_multi)

    df_result = df_best.copy()
    df_result = pd.concat([df_result, df_multi], axis=1)
    df_result['ofi_integrated'] = df_integrated
    return df_result


# Run it
if __name__ == "__main__":
    filepath = "first_25000_rows.csv"  # Adjust if needed
    df_ofi_all = compute_all_ofi_features(filepath)

    # Preview first few rows
    print(df_ofi_all.head())

