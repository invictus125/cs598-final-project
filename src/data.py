import csv
from functools import partial
from io import StringIO
import numpy as np
# from pathlib import Path
import vitaldb
import requests
from torch import FloatTensor, BoolTensor

SEGEMENT_LENGTH_SECONDS = 60

ABP_TRACK = 'SNUADC/ART'
ECG_TRACK = 'SNUADC/ECG_II'
EEG_TRACK = 'BIS/EEG1_WAV'

RELEVANT_TRACKS = [
    ABP_TRACK,
    ECG_TRACK,
    EEG_TRACK,
]

def _url_to_reader(url_string):
    response = requests.get(url_string)
    file = StringIO(response.text)
    return csv.DictReader(file, delimiter=',')

# def download_from_url(url_string, filepath):
#     with requests.get(url_string, stream=True) as response:
#         response.raise_for_status()
#         with open(filepath, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192): 
#                 file.write(chunk)

def get_unique_vals(key, iterable):
    return set(map(lambda item: item[key], iterable))

def case_filter(case):
    return float(case['age']) >= 18.0 and case['ane_type'] == 'General'

def _case_track_filter(case_id, case_dict):
    track_list = case_dict[case_id]['tracks']
    return (
        ABP_TRACK in track_list and
        ECG_TRACK in track_list and
        EEG_TRACK in track_list
    )

def _get_candidate_cases():
    cases_by_id = {}
    for case in _url_to_reader('https://api.vitaldb.net/cases'):
        if case_filter(case):
            case['tracks'] = {}
            cases_by_id[case['\ufeffcaseid']] = case

    track_list_reader = _url_to_reader('https://api.vitaldb.net/trks')

    for track in track_list_reader:
        case_id = track['caseid']
        if track['tname'] in RELEVANT_TRACKS:
            if cases_by_id.get(case_id):
                cases_by_id[case_id]['tracks'][track['tname']] = track['tid']

    case_track_filter = partial(_case_track_filter, case_dict=cases_by_id)

    return [case_id for case_id in filter(case_track_filter, cases_by_id.keys())], cases_by_id

def validate_abp_segment(segment):
    return (
        not np.isnan(segment).mean() > 0.1 and
        not (segment > 200).any() and
        not (segment < 30).any() and
        not ((np.max(segment) - np.min(segment)) < 30) and
        not (np.abs(np.diff(segment)) > 30).any() # abrupt changes are assumed to be noise
    )

# def download_data(path, max_num_candidate_cases=None):
#     candidate_case_ids, cases_by_id = _get_candidate_cases()

#     # verify no heart surgery in remaining cases
#     unique_operations = set(map(lambda key: cases_by_id[key]['optype'], candidate_case_ids))
#     print(unique_operations)

#     candidate_cases_downloaded = 0
#     for case_id in cases_by_id.keys():
#         print('Processing caseid: '+case_id)
#         case = cases_by_id[case_id]
        
#         for case_track in case['tracks']:
#             case_track_name = case_track['tname']
#             case_track_id = case_track['tid']
            
#             if case_track_name in ABP_TRACKS:
#                 track_type = 'ABP'
#             elif case_track_name in ECG_TRACKS:
#                 track_type = 'ECG'
#             else:
#                 track_type = 'EEG'

#             print('Getting data for track: ', case_track_name, 'with id:', case_track_id)

#             destination_path = path+f"/candidate_cases/{case_id}/{track_type}/{case_track_name.replace('/', '-')}-{case_track_id}.csv"
#             Path(destination_path[:destination_path.rfind('/')]).mkdir(parents=True, exist_ok=True)

#             download_from_url(
#                 'https://api.vitaldb.net/' + case_track_id,
#                 destination_path,
#             )
#             print('Data for track downloaded successfully')
        
#         candidate_cases_downloaded = candidate_cases_downloaded + 1

#         if max_num_candidate_cases is not None:
#             if candidate_cases_downloaded == max_num_candidate_cases:
#                 print('Requested number of cases reached.  ')
#                 break

def get_data(
    minutes_ahead,
    abp_and_ecg_sample_rate_per_second=500,
    eeg_sample_rate_per_second=128,
    max_num_samples=None,
    max_num_cases=None,
):
    candidate_case_ids, _ = _get_candidate_cases()
    abps = []
    ecgs = []
    eegs = []
    hypotension_event_bools = []

    abp_data_in_two_seconds = 2 * abp_and_ecg_sample_rate_per_second 

    case_count = 0
    for case_id in candidate_case_ids:
        case_num_samples = 0
        case_num_events = 0

        print('Getting track data for case:', case_id)
        case_tracks = vitaldb.load_case(int(case_id), RELEVANT_TRACKS[0:2], 1/abp_and_ecg_sample_rate_per_second)

        abp_track = case_tracks[:,0]
        # ecg_track = case_tracks[:,1]

        # eeg_track = vitaldb.load_case(int(case_id), RELEVANT_TRACKS[2], 1/eeg_sample_rate_per_second).flatten()

        for i in range(
            0,
            len(abp_track) - abp_and_ecg_sample_rate_per_second * (SEGEMENT_LENGTH_SECONDS + (1 + minutes_ahead) * SEGEMENT_LENGTH_SECONDS),
            10 * abp_and_ecg_sample_rate_per_second
        ):
            x_segment = abp_track[i:i + abp_and_ecg_sample_rate_per_second * SEGEMENT_LENGTH_SECONDS]
            y_segment_start = i + abp_and_ecg_sample_rate_per_second * (SEGEMENT_LENGTH_SECONDS + minutes_ahead * SEGEMENT_LENGTH_SECONDS)
            y_segement_end = i + abp_and_ecg_sample_rate_per_second * (SEGEMENT_LENGTH_SECONDS + (minutes_ahead + 1) * SEGEMENT_LENGTH_SECONDS)
            y_segment = abp_track[y_segment_start:y_segement_end]

            if validate_abp_segment(x_segment) and validate_abp_segment(y_segment):
                abps.append(x_segment)

                # 2 second moving average
                y_numerator = np.nancumsum(y_segment, dtype=np.float32)
                y_numerator[abp_data_in_two_seconds:] = y_numerator[abp_data_in_two_seconds:] - y_numerator[:-abp_data_in_two_seconds]
                y_moving_avg = y_numerator[abp_data_in_two_seconds - 1:] / abp_data_in_two_seconds

                is_hypotension_event = np.nanmax(y_moving_avg) < 65
                hypotension_event_bools.append(is_hypotension_event)
                case_num_samples = case_num_samples + 1
                if(is_hypotension_event):
                    case_num_events = case_num_events + 1
                print('Valid sample detected, is hypotension event?: ', is_hypotension_event)
            else:
                print('Invalid sample')

            at_max_samples = len(hypotension_event_bools) == max_num_samples
            if at_max_samples:
                break

        case_count = case_count + 1

        if at_max_samples or case_count == max_num_cases:
            if at_max_samples:
                print('Max samples reached')
            else:
                print('Max cases reached')
            break

    return (
        FloatTensor(np.array(abps)),
        FloatTensor(np.array(ecgs)),
        FloatTensor(np.array(eegs)),
        BoolTensor(np.array(hypotension_event_bools)),
    )



result = get_data(1, max_num_cases=2)
print(len(result[0]))
