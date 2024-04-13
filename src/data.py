import csv
from functools import partial
from io import StringIO
from pathlib import Path
import requests


EEG_TRACKS = frozenset([
    'BIS/EEG1_WAV',
    'BIS/EEG2_WAV',
])

ECG_TRACKS = frozenset([
    'SNUADC/ECG_II',
    'SNUADC/ECG_V5',
])

ABP_TRACKS = frozenset([
    'SNUADC/ART',
    'CardioQ/ABP',
])

RELEVANT_TRACKS = EEG_TRACKS.union(ECG_TRACKS).union(ABP_TRACKS)

def url_to_reader(url_string):
    response = requests.get(url_string)
    file = StringIO(response.text)
    return csv.DictReader(file, delimiter=',')

def get_unique_vals(key, iterable):
    return set(map(lambda item: item[key], iterable))

def case_filter(case):
    if float(case['age']) < 18.0 or (float(case['aneend']) - float(case['anestart']) == 0):
        return False
    else:
        return True

def track_filter(case_id, case_dict):
    tracks_set = set(
        map(
            lambda t: t['tname'],
            case_dict[case_id]['tracks']
        )
    )
    return (
        not tracks_set.isdisjoint(EEG_TRACKS) and
        not tracks_set.isdisjoint(ECG_TRACKS) and
        not tracks_set.isdisjoint(ABP_TRACKS)
    )

def is_valid_abp(abp_data):
    # Events and non-event cases extracted from noisy ABP waveforms with jSQI 0.8 or less, were excluded.
    # TODO: do some filtering based on SQI here
    return True

def preprocess_ecg_data(ecg_data):
    # TODO: do preprocessing
    # ECG waveforms were pre-processed with a 1–40-Hz band-pass filter and normalized using Z-score.
    processed_data = ecg_data
    return processed_data

def preprocess_eeg_data(eeg_data):
    # TODO: do preprocessing
    # EEG waveforms were pre-processed using a 0.5–50-Hz band-pass filter.
    processed_data = eeg_data
    return processed_data

def download_data(path):
    candidate_cases_by_id = {}

    for case in url_to_reader('https://api.vitaldb.net/cases'):
        if case_filter(case):
            case['tracks'] = []
            candidate_cases_by_id[case['\ufeffcaseid']] = case

    track_list_reader = url_to_reader('https://api.vitaldb.net/trks')

    for track in track_list_reader:
        case_id = track['caseid']
        if track['tname'] in RELEVANT_TRACKS:
            if candidate_cases_by_id.get(case_id):
                candidate_cases_by_id[case_id]['tracks'].append(track)

    case_track_filter = partial(track_filter, case_dict=candidate_cases_by_id)

    candidate_case_ids = [case_id for case_id in filter(case_track_filter, candidate_cases_by_id.keys())]

    qualified_case_ids =[]

    for case_id in candidate_case_ids:
        print('Processing case: '+case_id)
        case = candidate_cases_by_id[case_id]
        has_qualifying_abp_data = False
        case_tracks = {}
        for case_track in case['tracks']:
            case_track_name = case_track['tname']
            case_track_id = case_track['tid']
            print('Examining data for track: ', case_track_name, ' with id: ', case_track_id)
            track_reader = url_to_reader('https://api.vitaldb.net/' + case_track_id)
            track_data = []
            for track_row in track_reader:
                track_data.append(track_row)

            print('Data for track loaded successfully')

            if case_track_name in ABP_TRACKS:
                track_type = 'ABP'
                # ABP waveforms were used without pre-processing.
                if not has_qualifying_abp_data:
                    if is_valid_abp(track_data):
                        has_qualifying_abp_data = True

            elif case_track_name in ECG_TRACKS:
                track_type = 'ECG'
                track_data = preprocess_ecg_data(track_data)
            else:
                track_type = 'EEG'
                track_data = preprocess_eeg_data(track_data)

            case_tracks[f"{track_type}/{case_track_name.replace('/', '-')}-{case_track_id}.csv"] = track_data

        if(has_qualifying_abp_data):
            qualified_case_ids.append(case_id)
            for key in case_tracks.keys():
                track_data = case_tracks[key]
                headers = track_data[0].keys()
                filepath = path+f"/{case_id}/{key}"
                Path(filepath[:filepath.rfind('/')]).mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w') as track_file:
                    writer = csv.DictWriter(track_file, fieldnames=headers)
                    for td in track_data:
                        writer.writerow(td)

    # verify no heart surgery in remaining cases
    unique_operations = set(map(lambda key: candidate_cases_by_id[key]['optype'], qualified_case_ids))
    print(unique_operations)

    print(len(qualified_case_ids))

    # Waveform data track
    # The time column has three values: start time (0), time interval (s), and end time (s).
    # Time intervals between rows are assumed constant (monotonically increasing time).
    # Rows with missing values are not omitted, therefore the data is loadable as an array.

download_data('')