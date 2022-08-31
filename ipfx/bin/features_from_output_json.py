import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import allensdk.core.json_utilities as ju
import logging

def main():
    parser = argparse.ArgumentParser(
            description="Process feature json files from pipeline output into a flat csv."
        )
    parser.add_argument('files', type=str, nargs='+', help='feature json file(s) to process')
    parser.add_argument('--output', default='features.csv', help='path to write output csv')
    parser.add_argument('--qc', action='store_true', help='include qc and fx failure info in csv')
    args = parser.parse_args()
    process_file_list(args.files, output=args.output, save_qc_info=args.qc)


hero_sweep_features = [
    'adapt',
    'avg_rate',
    'latency',
    'first_isi',
    'mean_isi',
    "median_isi",
    "isi_cv",
    "ir_late"
]
rheo_sweep_features = [
    'latency',
    'avg_rate',
    'mean_isi',
    'first_isi',
]
mean_sweep_features = [
    'adapt',
    "isi_cv",
    "ir_late"
]
max_sweep_features = [
    # 'adapt',
    # "isi_cv",
    "avg_rate",
]
base_spike_features = [
    'upstroke_downstroke_ratio',
    'threshold_v',
    'peak_v',
    # not in cell record
    'width',
    'upstroke',
    'downstroke',
    'width_suprathresh',
]

ramp_spike_features = base_spike_features + [
    'trough_v',
    'threshold_i',
    'fast_trough_v',
    'postap_slope'
]
ls_spike_features = base_spike_features + [
    'trough_v',
    'fast_trough_v',
    'postap_slope'
]
rheo_last_spike_features = [
    'fast_trough_v',
    'adp_v',
]
spike_adapt_features = [
    'isi',
    'width',
    'upstroke',
    'downstroke',
    'threshold_v',
    'peak_v',
    # 'fast_trough_deltav',
]
invert_features = ["first_isi"]
spike_threshold_shift_features = ["fast_trough_v", "peak_v"]
sag_features = [
    'sag',
    # 'sag_peak_t',
    'sag_area',
    'sag_tau'
    ]
ls_features = sag_features + [
    "v_baseline",
    "rheobase_i",
    "fi_fit_slope",
    "vm_for_sag",
    "input_resistance",
    "input_resistance_ss",
    "tau",
]


def extract_pipeline_output(output_json, save_qc_info=False):
    output = ju.read(output_json)
    record = {}

    fx_output = output.get('feature_extraction', {})
    if save_qc_info:
        cell_qc_features = output.get("sweep_extraction", {}).get("cell_features")
        if cell_qc_features is not None:
            record.update({key+"_qc": val for key, val in cell_qc_features.items()})

        qc_state = output.get('qc', {}).get('cell_state')
        if qc_state is not None:
            record['fail_tags_qc'] = '; '.join(qc_state.pop('fail_tags'))
            record.update(qc_state)

        # fx_state = fx_output.get('cell_state')
        # if fx_state is not None:
        #     record['failed_fx'] = fx_state.get('failed_fx', False)
        #     record['fail_message_fx'] = fx_state.get('fail_fx_message')

        fx_state = fx_output.get('feature_states', {})
        for key, module_state in fx_state.items():
            name = key.replace('_state','')
            # record[f'fail_fx_message_{name}'] = module_state.get('fail_fx_message')
            record[f'failed_fx_{name}'] = module_state.get('failed_fx', False)

    cell_features = fx_output.get('cell_features', {})
    if cell_features is not None:
        record.update(extract_fx_output(cell_features))
    return record

sweep_qc_info = [
    "sampling_rate",
    "bridge_balance_mohm",
    "capacitance_compensation",
    "leak_pa",
    "pre_noise_rms_mv",
    "post_noise_rms_mv",
    "slow_noise_rms_mv",
    "vm_delta_mv",
]

def get_sweeps_df(output, features=[]):
    sweep_features = output.get('sweep_extraction', {}).get('sweep_features')
    assert sweep_features is not None and len(sweep_features) > 0
    sweep_df = pd.DataFrame.from_records(sweep_features).set_index('sweep_number')
    if len(features)>0:
        features.append('sweep_number')
        long_squares = (output.get('feature_extraction', {})
                    .get('cell_features', {})
                    .get('long_squares', {}))
        if long_squares is not None:
            ls_df = pd.DataFrame.from_records(long_squares['sweeps'], columns=features).set_index('sweep_number')
            sweep_df = sweep_df.join(ls_df)
    return sweep_df

def extract_sweep_qc_info(output_json, **kwargs):
    output = ju.read(output_json)
    record = {}
    sweep_df = get_sweeps_df(output)

    long_squares = (output.get('feature_extraction', {})
                    .get('cell_features', {})
                    .get('long_squares', {}))
    if long_squares is not None:
        for name in ['hero', 'rheobase']:
            number = long_squares.get(f'{name}_sweep', {}).get('sweep_number')
            if number is not None:
                sweep = sweep_df.loc[number]
                add_features_to_record(sweep_qc_info, sweep, record, suffix='_'+name)
    return record

def extract_fx_output(cell_features):
    record = {}

    ramps = cell_features.get('ramps')
    if ramps is not None:
        sweeps = ramps.get("spiking_sweeps", [])
        mean_spike_0 = get_mean_first_spike_features(sweeps, ramp_spike_features)
        add_features_to_record('all', mean_spike_0, record, suffix="_ramp")

    short_squares = cell_features.get('short_squares')
    if short_squares is not None:
        sweeps = short_squares.get("common_amp_sweeps", [])
        mean_spike_0 = get_mean_first_spike_features(sweeps, base_spike_features)
        add_features_to_record('all', mean_spike_0, record, suffix="_short_square")

    chirps = cell_features.get('chirps')
    if chirps is not None:
        add_features_to_record('all', chirps, record, suffix="_chirp")

    offset_feature_values(spike_threshold_shift_features, record, "threshold_v")
    invert_feature_values(invert_features, record)

    long_squares_analysis = cell_features.get('long_squares')
    if long_squares_analysis is not None:
        record.update(get_complete_long_square_features(long_squares_analysis))

    return record

def get_mean_first_spike_features(sweeps, features_list):
    record = {}
    spikes_sets = [sweep["spikes"] for sweep in sweeps]
    for feat in features_list:
        values = [ spikes[0][feat] for spikes in spikes_sets
               if len(spikes) > 0 and spikes[0][feat] is not None]
        record[feat] = np.nanmean(values) if len(values) > 0 else np.nan
    offset_feature_values(spike_threshold_shift_features, record, "threshold_v", replace=False)
    return record


def get_complete_long_square_features(long_squares_analysis):
    record = {}
    # include all scalar features
    # features = [feat for feat, val in long_squares_analysis.items() if np.isscalar(val)]
    add_features_to_record(ls_features, long_squares_analysis, record)

    if 'rheobase_sweep' in long_squares_analysis:
        sweep = long_squares_analysis.get('rheobase_sweep',{})
        add_features_to_record(rheo_sweep_features, sweep, record, suffix='_rheo')
        add_features_to_record(ls_spike_features, sweep["spikes"][0], record, suffix="_rheo")
        add_features_to_record(rheo_last_spike_features, sweep["spikes"][-1], record, suffix="_last_rheo")

    if 'hero_sweep' in long_squares_analysis:
        sweep = long_squares_analysis.get('hero_sweep',{})
        add_features_to_record(hero_sweep_features, sweep, record, suffix='_hero')
        add_features_to_record(ls_spike_features, sweep["spikes"][0], record, suffix="_hero")
        ahp_features = get_ahp_delay_ratio(sweep)
        add_features_to_record('all', ahp_features, record, suffix="_hero")

    if 'subthreshold_sweeps' in long_squares_analysis:
        sweeps = long_squares_analysis.get('subthreshold_sweeps',{})
        sweep_features_df = pd.DataFrame.from_records(sweeps)
        sweep = sweep_features_df.sort_values("stim_amp", ascending=False).iloc[0]
        if sweep["stim_amp"] > 0:
            add_features_to_record(sag_features, sweep, record, suffix='_depol')

    if 'spiking_sweeps' in long_squares_analysis:
        sweeps = long_squares_analysis.get('spiking_sweeps',{})
        sweep_features_df = pd.DataFrame.from_records(sweeps)
        mean_df = sweep_features_df[mean_sweep_features].mean(axis=0)
        add_features_to_record(mean_sweep_features, mean_df, record, suffix="_mean")
        max_df = sweep_features_df[max_sweep_features].max(axis=0)
        add_features_to_record(max_sweep_features, max_df, record, suffix="_max")

        # sweep = sweep_features_df.sort_values("stim_amp", ascending=False).iloc[0]
        # add_features_to_record(hero_sweep_features, sweep, record, suffix='_hero2')
        # add_features_to_record(ls_spike_features, sweep["spikes"][0], record, suffix="_hero2")

        n_adapt = 4
        spike_sets = [sweep["spikes"] for sweep in sweeps]
        # looking for one more spike than necessary here
        adapt_sweep = find_spiking_sweep_by_min_spikes(sweep_features_df, spike_sets, min_spikes=n_adapt+1)
        adapt_features = get_spike_adapt_ratio_features(spike_adapt_features, adapt_sweep, nth_spike=n_adapt)
        record.update(adapt_features)
        ahp_features = get_ahp_delay_ratio(adapt_sweep)
        add_features_to_record('all', ahp_features, record, suffix="_5spike")

        offset_feature_values(["trough_v"], record, "fast_trough_v", add_suffix="_slowdeltav", replace=False)
        offset_feature_values(spike_threshold_shift_features, record, "threshold_v", replace=False)
        invert_feature_values(invert_features, record)
    return record

def offset_feature_values(features, record, relative_to, add_suffix="_deltav", replace=True):
    for feature in features:
        matches = [x for x in record if x.startswith(feature)
                   and not 'adapt_ratio' in x and not 'last' in x]
        for match in matches:
            suffix = match[len(feature):]
            if record.get(relative_to+suffix) is not None:
                val = record[match]
                if replace:
                    record.pop(match)
                feature_short = feature.replace("_v", "")
                record[feature_short + add_suffix + suffix] = (val - record[relative_to+suffix]) if val is not None else None

def invert_feature_values(features, record):
    for feature in features:
        matches = [x for x in record if x.startswith(feature)]
        for match in matches:
            suffix = match[len(feature):]
            val = record.pop(match)
            record[feature + "_inv" + suffix] = 1/val if val is not None else None

def add_features_to_record(features, feature_data, record, suffix=""):
    if features is 'all':
        features = feature_data.keys()
    record.update({feature+suffix: feature_data.get(feature, np.nan) for feature in features})

def get_ahp_delay_ratio(sweep):
    spikes_set = sweep.get("spikes", [])
    if len(spikes_set) < 2:
        value = None
    else:
        isi = spikes_set[-1]['peak_t'] - spikes_set[-2]['peak_t']
        ahp = spikes_set[-2]['trough_t'] - spikes_set[-2]['peak_t']
        value = ahp/isi
    return {'ahp_delay_ratio': value}


def get_spike_adapt_ratio_features(features, sweep, nth_spike=4):
    spikes_set = sweep.get("spikes", [])
    suffix = '_adapt_ratio'
    record = {}
    nspikes = len(spikes_set)
    if 'isi' in features:
        for i in range(nspikes-1):
            spikes_set[i]['isi'] = spikes_set[i+1]['peak_t'] - spikes_set[i]['peak_t']
    for feature in features:
        if nspikes <= nth_spike:
            value = None
        else:
            nth = spikes_set[nth_spike-1].get(feature)
            first = spikes_set[0].get(feature)
            value = nth/first if (nth and first) else None
        record.update({feature+suffix: value})
    return record


def get_spikes_df(sweep_record, add_isi=True):
    df = pd.DataFrame.from_records(sweep_record['spikes'])
    if add_isi:
        df['isi'] = df['peak_t'].diff().shift(-1)
    return df

def get_spike_var_ratio_features(features, sweep):
    suffix='_var_ratio'
    spikes = get_spikes_df(sweep, add_isi='isi' in features)
    record = {feature+suffix: spikes[feature].max()/spikes[feature].min()
              for feature in features}
    return record

def find_spiking_sweep_by_min_spikes(spiking_features, spikes_set, min_spikes=5):
    num_spikes = np.array([len(spikes) for spikes in spikes_set])
    # spiking_features['spikes'] = spikes_set
    spiking_features = spiking_features.loc[num_spikes >= min_spikes].sort_values("stim_amp")
    spiking_features_depolarized = spiking_features[spiking_features["stim_amp"] > 0]

    if spiking_features_depolarized.empty:
        logging.info(f"Cannot find sweep with >={min_spikes} spikes.")
        return {}
    else:
        return spiking_features_depolarized.iloc[0]


def process_file_list(files, cell_ids=None, output=None, save_qc_info=False,
                      fcn=extract_pipeline_output):
    index_var = "cell_name"
    records = []
    for i, file in enumerate(files):
        record = fcn(file, save_qc_info=save_qc_info)
        if cell_ids is not None:
            record[index_var] = cell_ids[i]
        else:
        # use the parent folder for an id
        # could be smarter and check specimen_id vs name
            record[index_var] = Path(file).parent.name
        records.append(record)
    ephys_df = pd.DataFrame.from_records(records, index=index_var)
    if output:
        ephys_df.to_csv(output)
    return ephys_df

if __name__ == "__main__":
    main()

